import uuid, json
from kyc_provider import verify_identity
from db import InvestorDAO   # your SQLAlchemy / psycopg2 wrapper

def onboard_new_investor(form_data: dict, kyc_payload: dict, provider: str = 'persona'):
    """
    form_data – dict with investor personal details (name, email, etc.)
    kyc_payload – dict formatted for the chosen provider (see provider docs)
    """
    # 1️⃣ Run the KYC check
    kyc_result = verify_identity(kyc_payload, provider=provider)

    if kyc_result['status'] not in ('verified', 'APPROVED'):
        raise RuntimeError('KYC verification failed – cannot onboard investor.')

    # 2️⃣ Persist the investor record (store only the token hash!)
    investor = {
        'investor_id'   : str(uuid.uuid4()),
        'full_name'     : form_data['full_name'],
        'email'         : form_data['email'],
        'kyc_provider'  : kyc_result['provider'],
        'kyc_token_hash': kyc_result['token_hash'],
        'kyc_status'    : kyc_result['status'],
        'created_at'    : 'NOW()',
    }
    InvestorDAO.insert(investor)   # INSERT INTO investors (...)

    # 3️⃣ Return a friendly acknowledgement
    return {
        'investor_id': investor['investor_id'],
        'message'    : 'Investor onboarded successfully.'
    }
