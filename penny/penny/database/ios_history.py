"""Queue existing conversation history when introducing a device to iOS sync."""

SEED_IOS_HISTORY = """
    INSERT INTO ios_outbox (
        device_id, message_log_id, created_at, content, source_name,
        push_title, push_summary, legacy_id, attachments_json
    )
    SELECT registration.device_id, message.id, message.timestamp, '',
           COALESCE(
               (SELECT source_name FROM ios_outbox
                WHERE message_log_id = message.id AND source_name IS NOT NULL
                ORDER BY id LIMIT 1),
               message.mechanism
           ),
           '', '', 0,
           (SELECT attachments_json FROM ios_outbox
            WHERE message_log_id = message.id AND attachments_json IS NOT NULL
            ORDER BY id LIMIT 1)
    FROM ios_device_registration AS registration
    JOIN device ON device.id = registration.device_id
    CROSS JOIN messagelog AS message
    WHERE device.channel_type = 'ios' AND NOT message.is_reaction
      AND (:device_id IS NULL OR registration.device_id = :device_id)
    ON CONFLICT (device_id, message_log_id) DO NOTHING
"""
