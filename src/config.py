import os
from dotenv import load_dotenv
load_dotenv()


config = {
    "binance_keys": {
        "binanceaccount1": {"API_KEY": os.environ.get('BINANCE_API_KEY', ""), "SECRET_KEY": os.environ.get('BINANCE_SECRET_KEY', "")},
    },
    "bitmex_keys": {
        "bitmexaccount1": {"API_KEY": os.environ.get('BITMEX_API_KEY', ""), "SECRET_KEY": os.environ.get('BITMEX_SECRET_KEY', "")},
        "bitmexaccount2": {"API_KEY": "", "SECRET_KEY": ""}
    },
    "bitmex_test_keys": {
        "bitmextest1": {"API_KEY": "", "SECRET_KEY": ""},
        "bitmextest2": {"API_KEY": "", "SECRET_KEY": ""}
    },
    "line_apikey": {"API_KEY": ""}
}
