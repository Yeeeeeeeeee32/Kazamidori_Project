"""Japan Standard Time presentation; API/cache timestamps remain unambiguous."""
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9), name='JST')


def format_jst(value):
    if not value:
        return ''
    stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('A timezone is required')
    return stamp.astimezone(JST).strftime('%Y-%m-%d %H:%M JST')
