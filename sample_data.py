"""A realistic sample policy document so the demo works on a cold click.

Content is illustrative and modelled on a typical Indian savings-account
policy; it is not an actual IDBI Bank publication.
"""

from typing import Final

SAMPLE_DOC_NAME: Final[str] = "Savings Account Policy (Sample).txt"

SAMPLE_POLICY: Final[str] = """\
IDBI-STYLE SAVINGS ACCOUNT POLICY (ILLUSTRATIVE SAMPLE)

1. Minimum Balance Requirements
Metro and urban branch savings accounts require a Monthly Average Balance (MAB)
of Rs. 10,000. Semi-urban branches require a MAB of Rs. 5,000, and rural
branches require a MAB of Rs. 2,500. Salary accounts and Basic Savings Bank
Deposit Accounts (BSBDA) are exempt from any minimum balance requirement.

2. Non-Maintenance Charges
If the Monthly Average Balance falls below the required level, a non-maintenance
charge is levied based on the shortfall. The charge is Rs. 150 per month for
metro and urban accounts, and Rs. 100 per month for semi-urban and rural
accounts. Applicable taxes are added to all charges.

3. Interest Rate
Interest on savings accounts is calculated on the daily closing balance and
paid quarterly. The rate is 2.70% per annum for balances up to Rs. 10 lakh and
3.00% per annum for the amount above Rs. 10 lakh.

4. Debit Card and ATM Usage
A classic debit card carries an annual maintenance fee of Rs. 150 plus taxes,
waived in the first year. Customers get five free transactions per month at
other banks' ATMs in metro locations and unlimited free transactions at the
bank's own ATMs. Beyond the free limit, a fee of Rs. 21 per financial
transaction and Rs. 10 per non-financial transaction applies.

5. Cheque Book
The first cheque book of 20 leaves per financial year is free. Additional cheque
books are charged at Rs. 3 per leaf. Cheque book requests can be placed through
internet banking, the mobile app, or at any branch.

6. Know Your Customer (KYC) Requirements
Account opening requires one officially valid document for proof of identity and
one for proof of address. Accepted documents include Aadhaar, PAN card, passport,
voter ID, and driving licence. PAN or Form 60 is mandatory. Periodic KYC updation
is required every ten years for low-risk customers and every two years for
high-risk customers.

7. Nomination Facility
Nomination is strongly recommended and can be registered at account opening or
any time afterward at no charge. Only one nominee is permitted per savings
account. The nomination can be changed or cancelled by submitting the prescribed
form.

8. Account Closure
There is no charge for closing an account after 14 days and within 12 months of
opening if closed by the customer; a closure charge of Rs. 500 plus taxes applies
to accounts closed within 12 months. Accounts closed after 12 months are free of
closure charges. Any linked recurring or fixed deposits must be settled first.

9. Dormant and Inactive Accounts
An account with no customer-induced transaction for 24 months is classified as
inactive and then dormant. No charge is levied for reactivation. Reactivation
requires a fresh KYC verification and a written request at the home branch.

10. Grievance Redressal
Complaints may be raised through the 24x7 customer care number, the branch
manager, or the online grievance portal. If unresolved within 30 days, the
customer may escalate to the Banking Ombudsman under the Reserve Bank of India
Integrated Ombudsman Scheme.
"""
