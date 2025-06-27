# Bước 1: Nhập các thư viện cần thiết
import requests
import pandas as pd
import ta
import traceback
import logging
from fastapi import FastAPI, HTTPException

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bước 2: Định nghĩa ứng dụng FastAPI
app = FastAPI(
    title="Crypto Analyzer API",
    description="API để phân tích giá token từ Bitget theo thời gian thực"
)

# Bước 3: Sao chép hàm lấy dữ liệu nến từ tài liệu [cite: 5]
def get_bitget_klines(symbol: str, granularity: str, limit: int = 200):
    """
    Lấy dữ liệu nến (candlestick) từ API của Bitget.
    """
    try:
        logger.info(f"Calling Bitget API for {symbol} {granularity}")
        url = "https://api.bitget.com/api/v2/spot/market/candles"
        params = {
            "symbol": symbol.upper() + "USDT", # Tự động thêm USDT
            "granularity": granularity,
            "limit": limit
        }
        logger.info(f"Request params: {params}")
        
        response = requests.get(url, params=params)
        response.raise_for_status() # Báo lỗi nếu request không thành công
        data = response.json()
        
        logger.info(f"API response code: {data.get('code', 'N/A')}")
        
        if data['code'] != '00000': # Kiểm tra mã lỗi từ Bitget
            logger.error(f"Bitget API error: {data.get('msg', 'Unknown error')}")
            raise ValueError(data['msg'])
        return data['data']
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=503, detail=f"Lỗi khi gọi API Bitget: {e}")
    except (ValueError, KeyError) as e:
        logger.error(f"Data error: {e}")
        raise HTTPException(status_code=404, detail=f"Lỗi dữ liệu từ Bitget: {e}")


# Bước 4: Sao chép và mở rộng hàm phân tích kỹ thuật [cite: 6]
def analyze_data(candle_data: list):
    """
    Phân tích dữ liệu nến, tính toán các chỉ số kỹ thuật.
    """
    try:
        logger.info(f"Analyzing data with {len(candle_data) if candle_data else 0} candles")
        
        if not candle_data:
            logger.warning("No candle data received")
            return None

        df = pd.DataFrame(candle_data, columns=["timestamp", "open", "high", "low", "close", "baseVol", "quoteVol", "quoteVol2"])
        logger.info(f"DataFrame created with shape: {df.shape}")
        
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['quoteVol']) # Sử dụng quoteVol làm volume
        
        logger.info("Calculating technical indicators...")

        # Tính toán các chỉ báo
        rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
        macd = ta.trend.MACD(df['close']).macd().iloc[-1]
        macd_signal = ta.trend.MACD(df['close']).macd_signal().iloc[-1]
        
        # Tính toán các đường Moving Average
        ma10 = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator().iloc[-1]
        ma20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
        ma50 = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator().iloc[-1]

        latest_price = df['close'].iloc[-1]
        latest_volume = df['volume'].iloc[-1]

        result = {
            "latest_price": latest_price,
            "latest_volume": latest_volume,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "ma10": ma10,
            "ma20": ma20,
            "ma50": ma50
        }
        
        logger.info(f"Analysis completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {e}")
        logger.error(traceback.format_exc())
        raise

# Bước 5: Tạo API endpoint để Custom GPT có thể gọi
@app.get("/analyze")
def analyze_token(symbol: str):
    """
    Endpoint chính để phân tích token.
    GPT sẽ gọi endpoint này với một symbol (ví dụ: btc).
    API sẽ lấy dữ liệu các khung thời gian và trả về phân tích tổng hợp.
    """
    try:
        logger.info(f"Starting analysis for symbol: {symbol}")
        timeframes = ["5min", "15min", "30min", "1h", "4h"]
        analysis_results = {}

        for tf in timeframes:
            try:
                logger.info(f"Processing timeframe: {tf}")
                # Lấy dữ liệu nến
                kline_data = get_bitget_klines(symbol, tf)
                # Phân tích dữ liệu
                analysis = analyze_data(kline_data)
                analysis_results[tf] = analysis
            except HTTPException as e:
                # Nếu một khung thời gian lỗi, ghi nhận và tiếp tục
                logger.error(f"Error for timeframe {tf}: {e.detail}")
                analysis_results[tf] = {"error": e.detail}
            except Exception as e:
                logger.error(f"Unexpected error for timeframe {tf}: {e}")
                analysis_results[tf] = {"error": str(e)}

        logger.info("Analysis completed successfully")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Critical error in analyze_token: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Endpoint gốc để kiểm tra API có hoạt động không
@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API phân tích Crypto!"}