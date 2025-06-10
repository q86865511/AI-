"""
處理系統時區相關函數
"""
from datetime import datetime, timedelta, timezone

# 定義UTC+8時區
TW_TIMEZONE = timezone(timedelta(hours=8))

def get_tw_time() -> datetime:
    """
    獲取當前的台灣時間 (UTC+8)
    
    Returns:
        datetime: 表示當前台灣時間的datetime對象
    """
    return datetime.now(TW_TIMEZONE)

def format_tw_time(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化台灣時間
    
    Args:
        dt: 要格式化的datetime對象
        format_str: 格式化字符串
    
    Returns:
        str: 格式化後的時間字符串
    """
    # 如果沒有時區信息，添加台灣時區
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TW_TIMEZONE)
    return dt.strftime(format_str)

def to_tw_timezone(dt: datetime) -> datetime:
    """
    將任意時間轉換為台灣時區時間
    
    Args:
        dt: 任意時區的datetime對象
    
    Returns:
        datetime: 轉換為台灣時區的datetime對象
    """
    # 如果沒有時區信息，假定為UTC時間
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TW_TIMEZONE) 