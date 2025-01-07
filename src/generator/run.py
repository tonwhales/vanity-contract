import hashlib
from threading import Thread
import pyopencl as cl
import numpy as np
import base64
import time
import os
import argparse

def create_parser():
    parser = argparse.ArgumentParser('vanity-generator', description='Generate beautiful wallet for TON on GPU using vanity contract')
    parser.add_argument('owner', help='An owner of vanity contract')
    parser.add_argument('--end', type=str, default='', help='Search in the end of address, separated by , or |')
    parser.add_argument('--start', type=str, default='', help='Search in the start of address, separated by , or |')
    parser.add_argument('-nb', action='store_true', default=False, help='Search for non-bouncable addresses')
    parser.add_argument('-t', action='store_true', default=False, help='Search for testnet addresses')
    parser.add_argument('--threads', type=int, help='Worker threads')
    parser.add_argument('--its', type=int, default=10000, help='Worker iterations')
    parser.add_argument('-w', type=int, required=True, help='Address workchain')
    parser.add_argument('--case-sensitive', action='store_true', help='Search for case sensitive address (case insensitive by default)')
    parser.add_argument('--early-prefix', action='store_true', help='Check prefix starting from third character')
    parser.add_argument('--only-one', action='store_true', help='Stop when an address is found')
    return parser

def parse_parameters(param):
    return [p.strip() for p in param.replace('|', ',').split(',') if p.strip()]

def generate_condition_chunk(pattern, offset, case_sensitive):
    conditions = []
    for i, c in enumerate(pattern):
        if case_sensitive:
            condition = f"result[{offset + i}] == '{c}'"
        else:
            condition = f"(result[{offset + i}] == '{c}' || result[{offset + i}] == '{c.upper()}')"
        conditions.append(condition)
    return " && ".join(conditions)

def build_kernel_conditions(start_patterns, end_patterns, start_offset, case_sensitive):
    conditions = []

    # Split patterns into smaller chunks to avoid too long conditions
    MAX_PATTERNS_PER_CHUNK = 5

    # Handle start patterns
    if start_patterns:
        for i in range(0, len(start_patterns), MAX_PATTERNS_PER_CHUNK):
            chunk_patterns = start_patterns[i:i + MAX_PATTERNS_PER_CHUNK]
            chunk_conditions = [
                "(" + generate_condition_chunk(pattern, start_offset, case_sensitive) + ")"
                for pattern in chunk_patterns
            ]
            conditions.append("(" + " || ".join(chunk_conditions) + ")")

    # Handle end patterns
    if end_patterns:
        for i in range(0, len(end_patterns), MAX_PATTERNS_PER_CHUNK):
            chunk_patterns = end_patterns[i:i + MAX_PATTERNS_PER_CHUNK]
            chunk_conditions = [
                "(" + generate_condition_chunk(pattern, 48 - len(pattern), case_sensitive) + ")"
                for pattern in chunk_patterns
            ]
            conditions.append("(" + " || ".join(chunk_conditions) + ")")

    if not conditions:
        raise ValueError("No patterns provided")
    return " && ".join(conditions)

def crc16(data):
    reg = 0
    for b in data:
        mask = 0x80
        while mask > 0:
            reg <<= 1
            if b & mask:
                reg = reg + 1
            mask >>= 1
            if reg > 0xffff:
                reg = reg & 0xffff
                reg ^= 0x1021
    return reg.to_bytes(2, byteorder='big')

class VanityGenerator:
    def __init__(self, args, kernel_conditions):
        self.args = args
        self.kernel_conditions = kernel_conditions
        self.n_found = 0
        self.stopped = False
        self.owner_decoded = base64.urlsafe_b64decode(args.owner)
        self.inner_base = bytearray.fromhex('00840400') + self.owner_decoded[2:34]

        self.flags = (NON_BOUNCEABLE_TAG if args.nb else BOUNCEABLE_TAG)
        if args.t:
            self.flags |= TEST_FLAG
        self.flags <<= 8
        self.flags |= WORKCHAIN

        self.kernel_code = self.load_kernel_code()

    def load_kernel_code(self):
        kernel_path = os.path.join(os.path.dirname(__file__), 'vanity.cl')
        with open(kernel_path) as f:
            kernel_code = f.read()
        return kernel_code.replace("<<CONDITION>>", self.kernel_conditions)

    def solver(self, dev, context, queue, program):
        main = bytearray.fromhex('020134000100009b598624c569108630d69c8422af4b5971cd9d515ad83d4facec29e25b2f9c75d7c2f9ece11a5845e257cc6c8bd375459059902ce9f6206696a8964c5e7e078100')
        data = np.frombuffer(main, dtype=np.uint32)
        main_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 72, hostbuf=data)

        salt = bytearray(os.urandom(32))
        inner = self.inner_base + salt
        inner_data = np.frombuffer(inner, dtype=np.uint32)
        inner_g = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 68, hostbuf=inner_data)

        res = np.full(2048, 0xffffffff, np.uint32)
        res_g = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

        start = time.time()
        threads = self.args.threads or (dev.max_work_group_size * dev.max_compute_units)

        program.hash_main(
            queue,
            (threads,),
            None,
            np.int32(self.args.its),
            np.ushort(self.flags),
            np.int32(71),
            main_g,
            np.int32(68),
            inner_g,
            res_g
        ).wait()

        result = np.empty(2048, np.uint32)
        cl.enqueue_copy(queue, result, res_g).wait()

        self.process_results(result, salt, main)

        speed = round(threads * self.args.its / (time.time() - start) / 1e6)
        print(f'Speed: {speed} Mh/s, found: {self.n_found}')

    def process_results(self, result, salt, main):
        ps = list(np.where(result != 0xffffffff))[0]
        for j in range(0, len(ps), 2):
            p = ps[j]
            a = result[p]
            b = result[p+1]

            salt_np = np.frombuffer(salt, np.uint32)
            salt_np[0] ^= a
            salt_np[1] ^= b
            hdata1 = self.inner_base + salt_np.tobytes()
            hash1 = hashlib.sha256(hdata1).digest()
            main[39:71] = hash1

            hs = hashlib.sha256(main[:71]).digest()
            address = self.create_address(hs)

            if self.is_valid_address(address):
                self.save_found_address(address, salt_np)
                if self.args.only_one:
                    self.stopped = True
                    os._exit(0)

    def create_address(self, hs):
        address = bytearray()
        address += self.flags.to_bytes(2, 'big')
        address += hs
        address += b'\x00\x00'
        crc = crc16(address)
        address[34] = crc[0]
        address[35] = crc[1]
        return base64.urlsafe_b64encode(address).decode('utf-8')

    def is_valid_address(self, address):
        return any(address.lower().endswith(pattern) for pattern in self.end_patterns) or \
               any(address[self.start_offset:].lower().startswith(pattern) for pattern in self.start_patterns)

    def save_found_address(self, address, salt_np):
        print(f'Found: {address} salt: {salt_np.tobytes().hex()}')
        with open('found.txt', 'a') as f:
            f.write(f'{address} {salt_np.tobytes().hex()}\n')
        self.n_found += 1

    def run(self):
        platforms = cl.get_platforms()
        threads = []

        for platform in platforms:
            devices = platform.get_devices(cl.device_type.GPU)
            for _, dev in enumerate(devices, 1):
                print(f"Using device: {dev.name}")
                t = Thread(target=self.device_thread, args=(dev,))
                threads.append(t)
                t.start()

        try:
            while not self.stopped:
                [t.join(1) for t in threads]
        except KeyboardInterrupt:
            print('Interrupted')
            self.stopped = True
            os._exit(0)

    def device_thread(self, device):
        context = cl.Context(devices=[device])
        queue = cl.CommandQueue(context)
        program = cl.Program(context, self.kernel_code).build()

        while not self.stopped:
            self.solver(device, context, queue, program)

def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.end and not args.start:
        parser.print_usage()
        print('vanity-generator: error: the following arguments are required: end or start')
        os._exit(0)

    # Constants
    global BOUNCEABLE_TAG, NON_BOUNCEABLE_TAG, TEST_FLAG, WORKCHAIN, mf
    BOUNCEABLE_TAG = 0x11
    NON_BOUNCEABLE_TAG = 0x51
    TEST_FLAG = 0x80
    WORKCHAIN = (args.w + (1 << 8)) % (1 << 8)
    mf = cl.mem_flags

    # Parse patterns
    start_patterns = parse_parameters(args.start)
    end_patterns = parse_parameters(args.end)

    if not args.case_sensitive:
        start_patterns = [p.lower() for p in start_patterns]
        end_patterns = [p.lower() for p in end_patterns]

    start_offset = 2 if args.early_prefix else 3

    # Build kernel conditions
    kernel_conditions = build_kernel_conditions(
        start_patterns,
        end_patterns,
        start_offset,
        args.case_sensitive
    )

    # Print configuration
    print("\nConfiguration:")
    print(f"Owner: {args.owner}")
    print(f"Start patterns: {start_patterns}")
    print(f"End patterns: {end_patterns}")
    print(f"Case sensitive: {args.case_sensitive}")
    print(f"Early prefix: {args.early_prefix}")
    print(f"Kernel conditions: {kernel_conditions}\n")

    generator = VanityGenerator(args, kernel_conditions)
    generator.start_patterns = start_patterns
    generator.end_patterns = end_patterns
    generator.start_offset = start_offset
    generator.run()

if __name__ == "__main__":
    main()