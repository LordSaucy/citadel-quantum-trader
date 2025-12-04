def log_action(action: str, user_id: str, status: str, details: dict = None):
    asset = details.get("asset_class", "unknown")
    session = details.get("session", "none")
    # Include asset & session in the hash payload
    merkle_input = f"{action}{user_id}{status}{asset}{session}{json.dumps(details, sort_keys=True)}"
    ...
    cursor.execute(
        "INSERT INTO admin_audit_log (action,user_id,status,details,merkle_hash,previous_hash,asset_class,session) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (action, user_id, status, json.dumps(details or {}), merkle_hash,
         previous_hash, asset, session)
    )
