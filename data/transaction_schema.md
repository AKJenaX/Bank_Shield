# Transaction Schema

Required keys for each transaction record:

- `id` or `transaction_id` (string)
- `amount` (number)
- `true_label` (`fraud` or `normal`)

Recommended optional keys:

- `merchant`
- `category`
- `timestamp`
- `metadata` (object)
