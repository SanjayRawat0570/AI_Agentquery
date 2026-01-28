from main import ACCOUNTS, update_account_subscription
import json

acct_id = 'ACC-1111'
print('Before:')
print(json.dumps(ACCOUNTS.get(acct_id), indent=2))

print('\nApplying subscription change to cm-enterprise...')
updated = update_account_subscription(acct_id, 'cm-enterprise')
print('\nAfter:')
print(json.dumps(updated, indent=2))
