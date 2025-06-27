# Bước 1: Nhập các thư viện cần thiết
import requests
import pandas as pd
import ta
import traceback
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bước 2: Định nghĩa các model Pydantic cho response
class IndicatorsModel(BaseModel):
    latest_price: float = Field(..., description="Giá hiện tại của token")
    latest_volume: float = Field(..., description="Volume giao dịch hiện tại")
    volume_avg_20: Optional[float] = Field(None, description="Volume trung bình 20 periods")
    ema10: Optional[float] = Field(None, description="Exponential Moving Average 10 periods")
    ema20: Optional[float] = Field(None, description="Exponential Moving Average 20 periods")
    ema50: Optional[float] = Field(None, description="Exponential Moving Average 50 periods")
    ema200: Optional[float] = Field(None, description="Exponential Moving Average 200 periods")

class GTICriteriaModel(BaseModel):
    trend_condition_met: bool = Field(..., description="Điều kiện trend của hệ thống GTI")
    price_above_ema10: bool = Field(..., description="Giá có trên EMA10")
    price_above_ema20: bool = Field(..., description="Giá có trên EMA20")
    ema10_above_ema20: bool = Field(..., description="EMA10 có trên EMA20 (uptrend)")
    volume_breakout_on_latest_candle: bool = Field(..., description="Có breakout volume không")
    pullback_to_ema10: bool = Field(..., description="Có pullback về EMA10")
    pullback_to_ema20: bool = Field(..., description="Có pullback về EMA20")
    note: str = Field(..., description="Ghi chú về điều kiện trend")

class OHLCModel(BaseModel):
    timestamp: int = Field(..., description="Timestamp của nến")
    open: float = Field(..., description="Giá mở")
    high: float = Field(..., description="Giá cao nhất")
    low: float = Field(..., description="Giá thấp nhất")
    close: float = Field(..., description="Giá đóng")
    volume: float = Field(..., description="Volume giao dịch")

class DataQualityModel(BaseModel):
    total_candles: int = Field(..., description="Tổng số nến")
    valid_candles: int = Field(..., description="Số nến hợp lệ")
    timeframe: str = Field(..., description="Khung thời gian")

class TimeframeAnalysisModel(BaseModel):
    indicators: Optional[IndicatorsModel] = Field(None, description="Các chỉ báo kỹ thuật")
    gti_criteria_checks: Optional[GTICriteriaModel] = Field(None, description="Kiểm tra tiêu chí GTI")
    ohlc_data: Optional[List[OHLCModel]] = Field(None, description="Dữ liệu OHLC")
    data_quality: Optional[DataQualityModel] = Field(None, description="Chất lượng dữ liệu")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")

class AnalysisResponseModel(BaseModel):
    symbol: str = Field(..., description="Symbol của token được phân tích")
    analysis_results: Dict[str, TimeframeAnalysisModel] = Field(..., description="Kết quả phân tích theo từng timeframe")
    timestamp: str = Field(..., description="Thời gian thực hiện phân tích")

# Bước 3: Định nghĩa ứng dụng FastAPI với OpenAPI config
app = FastAPI(
    title="Crypto GTI Analyzer API",
    description="API phân tích cryptocurrency theo hệ thống GTI với dữ liệu từ Bitget. Hỗ trợ phân tích đa timeframe và các chỉ báo kỹ thuật.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Bước 4: Hàm lấy dữ liệu nến từ Bitget
def get_bitget_klines(symbol: str, granularity: str, limit: int = 200):
    """Lấy dữ liệu nến (candlestick) từ API của Bitget."""
    try:
        # Ánh xạ khung thời gian cho API Bitget
        granularity_map = {
            "1W": "1week", 
            "1D": "1day", 
            "4h": "4h", 
            "1h": "1h", 
            "30min": "30min", 
            "15min": "15min",
            "5min": "5min"
        }
        api_granularity = granularity_map.get(granularity, granularity)

        logger.info(f"Calling Bitget API for {symbol} on timeframe {granularity} (API: {api_granularity})")
        url = "https://api.bitget.com/api/v2/spot/market/candles"
        params = {"symbol": symbol.upper() + "USDT", "granularity": api_granularity, "limit": limit}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] != '00000':
            logger.error(f"Bitget API error: {data.get('msg', 'Unknown error')}")
            raise ValueError(data['msg'])
        
        if not data.get('data') or len(data['data']) == 0:
            raise ValueError("No data returned from API")
        
        return data['data']
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=503, detail=f"Lỗi khi gọi API Bitget: {e}")
    except (ValueError, KeyError) as e:
        logger.error(f"Data error: {e}")
        raise HTTPException(status_code=404, detail=f"Lỗi dữ liệu từ Bitget: {e}")


# Bước 4: Hàm phân tích kỹ thuật và kiểm tra tiêu chí GTI
def analyze_data(candle_data: list, timeframe: str):
    """Phân tích dữ liệu nến, tính toán chỉ số và kiểm tra các tiêu chí của hệ thống GTI."""
    try:
        if not candle_data or len(candle_data) < 50:
            logger.warning(f"Not enough candle data for analysis on {timeframe} ({len(candle_data) if candle_data else 0} candles)")
            return {"error": "Không đủ dữ liệu để phân tích."}

        # Kiểm tra format dữ liệu từ API
        logger.info(f"Sample data structure: {candle_data[0] if candle_data else 'No data'}")
        
        # Tạo DataFrame với proper error handling
        try:
            df = pd.DataFrame(candle_data)
            
            # Kiểm tra số cột và định nghĩa tên cột phù hợp
            if df.shape[1] >= 7:
                df.columns = ["timestamp", "open", "high", "low", "close", "baseVol", "usdtVol"] + [f"col_{i}" for i in range(7, df.shape[1])]
            else:
                logger.error(f"Insufficient columns in data: {df.shape[1]} columns, expected at least 7")
                return {"error": f"Dữ liệu không đúng format: {df.shape[1]} cột"}
            
        except Exception as e:
            logger.error(f"DataFrame creation error: {e}")
            return {"error": f"Lỗi tạo DataFrame: {e}"}
        
        # Convert với error handling tốt hơn
        numeric_columns = ['open', 'high', 'low', 'close', 'baseVol', 'usdtVol']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.error(f"Missing column: {col}")
                return {"error": f"Thiếu cột dữ liệu: {col}"}
        
        # Kiểm tra NaN values
        if df[numeric_columns].isnull().any().any():
            logger.warning("Found NaN values in data, filling with forward fill")
            df[numeric_columns] = df[numeric_columns].ffill()
        
        # Sử dụng usdtVol làm volume chính
        df['volume'] = df['usdtVol']
        
        # Đảm bảo có đủ dữ liệu hợp lệ
        if len(df.dropna()) < 20:
            return {"error": "Quá nhiều dữ liệu không hợp lệ sau khi xử lý"}
        
        # --- TÍNH TOÁN CÁC CHỈ BÁO với error handling ---
        try:
            # EMA ngắn hạn
            ema10 = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
            ema20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            
            # Lấy giá trị cuối cùng, kiểm tra NaN
            ema10_val = ema10.iloc[-1] if not pd.isna(ema10.iloc[-1]) else None
            ema20_val = ema20.iloc[-1] if not pd.isna(ema20.iloc[-1]) else None
            
            # EMA dài hạn chỉ cho khung Ngày và Tuần
            ema50_val, ema200_val = None, None
            if timeframe in ["1D", "1W"] and len(df) >= 200:
                ema50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
                ema200 = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
                ema50_val = ema50.iloc[-1] if not pd.isna(ema50.iloc[-1]) else None
                ema200_val = ema200.iloc[-1] if not pd.isna(ema200.iloc[-1]) else None
            
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return {"error": f"Lỗi tính toán EMA: {e}"}
        
        # Volume analysis với protection
        try:
            volume_avg_20 = df['volume'].rolling(window=20).mean().iloc[-1]
            if pd.isna(volume_avg_20) or volume_avg_20 <= 0:
                volume_avg_20 = df['volume'].mean()
        except Exception as e:
            logger.warning(f"Volume calculation issue: {e}")
            volume_avg_20 = df['volume'].mean() if not df['volume'].empty else 1
        
        # Lấy dữ liệu mới nhất
        latest_price = float(df['close'].iloc[-1])
        latest_volume = float(df['volume'].iloc[-1])
        latest_low = float(df['low'].iloc[-1])
        latest_high = float(df['high'].iloc[-1])

        # --- KIỂM TRA CÁC TIÊU CHÍ GTI với protection ---
        gti_results = {}
        
        # Kiểm tra trend condition chỉ khi có đủ EMA data
        if ema10_val is not None and ema20_val is not None:
            price_above_emas = latest_price > ema10_val and latest_price > ema20_val
            gti_trend_condition = (ema10_val > ema20_val) and price_above_emas
            gti_results.update({
                "trend_condition_met": bool(gti_trend_condition),
                "price_above_ema10": bool(latest_price > ema10_val),
                "price_above_ema20": bool(latest_price > ema20_val),
                "ema10_above_ema20": bool(ema10_val > ema20_val)
            })
        else:
            gti_results.update({
                "trend_condition_met": False,
                "error": "Không đủ dữ liệu để tính EMA"
            })

        # Kiểm tra volume breakout
        volume_breakout = bool(latest_volume > (1.5 * volume_avg_20)) if volume_avg_20 > 0 else False
        
        # Kiểm tra pullback
        is_pullback_to_ema10 = False
        is_pullback_to_ema20 = False
        
        if ema10_val is not None:
            is_pullback_to_ema10 = bool(latest_low <= ema10_val and latest_price > ema10_val)
        if ema20_val is not None:
            is_pullback_to_ema20 = bool(latest_low <= ema20_val and latest_price > ema20_val)

        # Lấy OHLC data với protection
        try:
            ohlc_data = []
            for i, row in df.tail(min(100, len(df))).iterrows():
                # Convert all values to native Python types to avoid numpy serialization issues
                ohlc_data.append({
                    "timestamp": int(float(row['timestamp'])),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                })
        except Exception as e:
            logger.error(f"OHLC data processing error: {e}")
            ohlc_data = []

        result = {
            "indicators": {
                "latest_price": float(latest_price),
                "latest_volume": float(latest_volume),
                "volume_avg_20": float(volume_avg_20) if volume_avg_20 is not None else None,
                "ema10": float(ema10_val) if ema10_val is not None else None,
                "ema20": float(ema20_val) if ema20_val is not None else None,
                "ema50": float(ema50_val) if ema50_val is not None else None,
                "ema200": float(ema200_val) if ema200_val is not None else None
            },
            "gti_criteria_checks": {
                **gti_results,
                "volume_breakout_on_latest_candle": bool(volume_breakout),
                "pullback_to_ema10": bool(is_pullback_to_ema10),
                "pullback_to_ema20": bool(is_pullback_to_ema20),
                "note": "Trend condition: EMA10 > EMA20 (uptrend) + price above both EMAs"
            },
            "ohlc_data": ohlc_data,
            "data_quality": {
                "total_candles": int(len(df)),
                "valid_candles": int(len(df.dropna())),
                "timeframe": timeframe
            }
        }
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_data on {timeframe}: {e}\n{traceback.format_exc()}")
        return {"error": f"Lỗi phân tích trên khung {timeframe}: {str(e)}"}

# Bước 5: Tạo API endpoint
@app.get("/analyze", response_model=AnalysisResponseModel, 
         summary="Phân tích crypto theo hệ thống GTI",
         description="Phân tích token cryptocurrency trên nhiều timeframe với các chỉ báo kỹ thuật và tiêu chí GTI")
def analyze_token(
    symbol: str = Query(..., 
                       description="Symbol của cryptocurrency cần phân tích (ví dụ: BTC, ETH, ADA)", 
                       example="BTC")
):
    """
    Phân tích cryptocurrency theo hệ thống GTI.
    
    - **symbol**: Symbol của token (VD: BTC, ETH, ADA)
    - Trả về phân tích trên 7 timeframes: 1W, 1D, 4h, 1h, 30min, 15min, 5min
    - Bao gồm các chỉ báo: EMA, Volume, GTI criteria
    """
    try:
        logger.info(f"Starting GTI analysis for symbol: {symbol}")
        # Cập nhật timeframes để bao gồm 1D và 1W
        timeframes = ["1W", "1D", "4h", "1h", "30min", "15min", "5min"]
        analysis_results = {}

        for tf in timeframes:
            try:
                logger.info(f"Processing timeframe: {tf}")
                kline_data = get_bitget_klines(symbol, tf, limit=250)
                analysis = analyze_data(kline_data, tf)
                analysis_results[tf] = analysis
                logger.info(f"Completed analysis for {tf}")
            except Exception as e:
                logger.error(f"Error processing timeframe {tf}: {e}")
                analysis_results[tf] = {"error": f"Lỗi xử lý khung thời gian {tf}: {str(e)}"}

        return {
            "symbol": symbol.upper(),
            "analysis_results": analysis_results,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Critical error in analyze_token: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Endpoint gốc
@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API phân tích Crypto tích hợp hệ thống GTI!"}