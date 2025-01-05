# TON Vanity Contract

Smart contract that allows to "mine" any suitable address for any contract. One of the nice side-effects is that you can also use it as a primary deployment method for your contracts since you are specifying data and code as is and there are no need to have an alternative way init contract.

* 🔐 Secure. Address allocation binded to the wallet you control.
* 🚀 Fast. Suffix with few letters takes only few seconds.
* 🤝 Convenient. Vanity Contract is a simple way to deploy contracts and allows avoid implementing separate deployment logic.

## How it works

Creating a vanity address is a two step process:

1) "Mine" an addres with a our python tool. You can mine 5 letter suffix in seconds.
2) Deploy your contract code and data using special vanity message


## Mining Address

Checkout this repository on your machine with `Python 3.9` installed. It is recommended to use some computer with good GPU.

Execute in `src/generator` and replace `<suffix>,<suffix2>` with desired suffix names and `<owner>` with an address of a wallet that then could be used for deployment.

```bash
python3.9 run.py --end '<suffix>,<suffix2>' -w -1 --case-sensitive <owner>
```

Example output:
```bash
...
Found:  Ef_q7x1ZcJMZcWhanI6-t3giwAHOo6iX2YgovTvk0GryCLUB salt:  04fd8dda6b23067c1aa45c5af500174dc6c4d79b7d50c9de81ffb9e4a62c2d2a
Found:  Ef9Qa5dP5Y4E6xBURdBBF8_P8XMTdLJaQLGkoUA5Eya2CLUB salt:  b0181470628ea0c341c4327e188dfba1a46be8527933f031ef61af6f35ffacf8
...
```

Store this salt and an address somwhere for future deployment.

## Deployment

Deployment is performed by sending an internal message with desired code and data.

Code example how to deploy:

```ts

import qs from 'qs';
import { Address, beginCell, Cell } from 'ton';

const salt = '<salt>';
const owner = Address.parse('<owner>');
const targetAddress = Address.parse('<address>');
const testnet = true;

// Vanity contract code and data
const vanityCode = Cell.fromBoc(
  Buffer.from('te6ccgEBAgEAMgABFP8A9KQT9LzyyAsBAEbT7UTQddch+kCDB9ch0QLQ0wMx+kAwWMcF8ojU1NEB+wTtVA==')
)[0];
const vanityData = beginCell()
  .storeUint(0, 5)
  .storeAddress(owner)
  .storeBuffer(Buffer.from(salt, 'hex'))
  .endCell();
let msg = new StateInit({
  code: vanityCode,
  data: vanityData,
});
let cell = new Cell();
msg.writeTo(cell);
let vanityInit = cell.toBoc({ idx: false }).toString("base64");

// Your contract code and data
const code = Cell.fromBoc('....');
const data = Cell.fromBoc('....');
const deployMessage = beginCell()
  .storeRef(code)
  .storeRef(data)
  .toBoc({ idx: false })
  .toString("base64");

// Create and display link
let link = `https://${testnet ? 'test.' : ''}tonhub.com/transfer/` + targetAddress.toFriendly({ testOnly: testnet }) + "?" + qs.stringify({
  text: "Deploy contract",
  amount: amount.toString(1),
  init: vanityInit,
  bin: deployMessage,
});
console.log("Deploy: " + link);

```

## License

MIT
