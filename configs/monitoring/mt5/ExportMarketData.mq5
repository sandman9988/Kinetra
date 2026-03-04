//+------------------------------------------------------------------+
//|                                              ExportMarketData.mq5 |
//|                                                          Kinetra |
//|                         Exports MarketWatch data to CSV files    |
//+------------------------------------------------------------------+
#property copyright "Kinetra"
#property link      "https://github.com/kinetra"
#property version   "2.00"
#property description "Export market data with full MQL5 compliance"
#property script_show_inputs
#property strict

//--- Input parameters
input int      DaysToExport = 365;           // Days of history
input string   ExportPath = "KinetraData";   // Export folder name (in MQL5/Files/)
input bool     ExportM15 = true;             // Export M15
input bool     ExportM30 = true;             // Export M30
input bool     ExportH1 = true;              // Export H1
input bool     ExportH4 = true;              // Export H4
input bool     ExportD1 = false;             // Export D1
input bool     ExportSymbolInfo = true;      // Export symbol specifications
input bool     IncludeSpread = true;         // Include spread in OHLCV data
input bool     IncludeRealVolume = false;    // Include real volume (if available)

//+------------------------------------------------------------------+
//| Get timeframe as string (MQL5 standard)                          |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M2:  return "M2";
      case PERIOD_M3:  return "M3";
      case PERIOD_M4:  return "M4";
      case PERIOD_M5:  return "M5";
      case PERIOD_M6:  return "M6";
      case PERIOD_M10: return "M10";
      case PERIOD_M12: return "M12";
      case PERIOD_M15: return "M15";
      case PERIOD_M20: return "M20";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H2:  return "H2";
      case PERIOD_H3:  return "H3";
      case PERIOD_H4:  return "H4";
      case PERIOD_H6:  return "H6";
      case PERIOD_H8:  return "H8";
      case PERIOD_H12: return "H12";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default: return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| Get timeframe in minutes (for calculations)                       |
//+------------------------------------------------------------------+
int TimeframeToMinutes(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return 1;
      case PERIOD_M2:  return 2;
      case PERIOD_M3:  return 3;
      case PERIOD_M4:  return 4;
      case PERIOD_M5:  return 5;
      case PERIOD_M6:  return 6;
      case PERIOD_M10: return 10;
      case PERIOD_M12: return 12;
      case PERIOD_M15: return 15;
      case PERIOD_M20: return 20;
      case PERIOD_M30: return 30;
      case PERIOD_H1:  return 60;
      case PERIOD_H2:  return 120;
      case PERIOD_H3:  return 180;
      case PERIOD_H4:  return 240;
      case PERIOD_H6:  return 360;
      case PERIOD_H8:  return 480;
      case PERIOD_H12: return 720;
      case PERIOD_D1:  return 1440;
      case PERIOD_W1:  return 10080;
      case PERIOD_MN1: return 43200;
      default: return 60;
   }
}

//+------------------------------------------------------------------+
//| Get trade mode as string                                          |
//+------------------------------------------------------------------+
string TradeModeToString(ENUM_SYMBOL_TRADE_MODE mode)
{
   switch(mode)
   {
      case SYMBOL_TRADE_MODE_DISABLED:  return "DISABLED";
      case SYMBOL_TRADE_MODE_LONGONLY:  return "LONG_ONLY";
      case SYMBOL_TRADE_MODE_SHORTONLY: return "SHORT_ONLY";
      case SYMBOL_TRADE_MODE_CLOSEONLY: return "CLOSE_ONLY";
      case SYMBOL_TRADE_MODE_FULL:      return "FULL";
      default: return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| Get swap mode as string                                           |
//+------------------------------------------------------------------+
string SwapModeToString(ENUM_SYMBOL_SWAP_MODE mode)
{
   switch(mode)
   {
      case SYMBOL_SWAP_MODE_DISABLED:        return "DISABLED";
      case SYMBOL_SWAP_MODE_POINTS:          return "POINTS";
      case SYMBOL_SWAP_MODE_CURRENCY_SYMBOL: return "CURRENCY_SYMBOL";
      case SYMBOL_SWAP_MODE_CURRENCY_MARGIN: return "CURRENCY_MARGIN";
      case SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT: return "CURRENCY_DEPOSIT";
      case SYMBOL_SWAP_MODE_INTEREST_CURRENT: return "INTEREST_CURRENT";
      case SYMBOL_SWAP_MODE_INTEREST_OPEN:   return "INTEREST_OPEN";
      case SYMBOL_SWAP_MODE_REOPEN_CURRENT:  return "REOPEN_CURRENT";
      case SYMBOL_SWAP_MODE_REOPEN_BID:      return "REOPEN_BID";
      default: return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| Get day of week as string                                         |
//+------------------------------------------------------------------+
string DayOfWeekToString(ENUM_DAY_OF_WEEK day)
{
   switch(day)
   {
      case SUNDAY:    return "SUNDAY";
      case MONDAY:    return "MONDAY";
      case TUESDAY:   return "TUESDAY";
      case WEDNESDAY: return "WEDNESDAY";
      case THURSDAY:  return "THURSDAY";
      case FRIDAY:    return "FRIDAY";
      case SATURDAY:  return "SATURDAY";
      default: return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| Export symbol data to CSV                                         |
//+------------------------------------------------------------------+
bool ExportSymbolData(string symbol, ENUM_TIMEFRAMES timeframe, string folder)
{
   // Calculate date range
   datetime endDate = TimeCurrent();
   datetime startDate = endDate - DaysToExport * 24 * 60 * 60;
   
   // Request data from server (ensures fresh data)
   if(!SymbolSelect(symbol, true))
   {
      Print("Failed to select symbol: ", symbol);
      return false;
   }

   // Get rates using MQL5 CopyRates function
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(symbol, timeframe, startDate, endDate, rates);

   if(copied <= 0)
   {
      int error = GetLastError();
      Print("No data for ", symbol, " ", TimeframeToString(timeframe), 
            " Error: ", error, " (", ErrorDescription(error), ")");
      return false;
   }

   // Generate filename with date range
   string tfStr = TimeframeToString(timeframe);
   string startStr = TimeToString(rates[copied-1].time, TIME_DATE);
   string endStr = TimeToString(rates[0].time, TIME_DATE);
   StringReplace(startStr, ".", "");
   StringReplace(endStr, ".", "");
   
   // Format: SYMBOL_TF_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.csv
   string startFull = TimeToString(rates[copied-1].time, TIME_DATE|TIME_MINUTES);
   string endFull = TimeToString(rates[0].time, TIME_DATE|TIME_MINUTES);
   StringReplace(startFull, ".", "");
   StringReplace(startFull, ":", "");
   StringReplace(startFull, " ", "");
   StringReplace(endFull, ".", "");
   StringReplace(endFull, ":", "");
   StringReplace(endFull, " ", "");

   string filename = folder + "/" + symbol + "_" + tfStr + "_" + startFull + "_" + endFull + ".csv";

   // Open file for writing
   int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      int error = GetLastError();
      Print("Failed to create file: ", filename, " Error: ", error);
      return false;
   }

   // Write header based on options
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   
   if(IncludeSpread && IncludeRealVolume)
      FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume");
   else if(IncludeSpread)
      FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume", "spread");
   else if(IncludeRealVolume)
      FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume", "real_volume");
   else
      FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume");

   // Write data (oldest first - reverse from series order)
   for(int i = copied - 1; i >= 0; i--)
   {
      string timeStr = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);
      
      if(IncludeSpread && IncludeRealVolume)
      {
         FileWrite(handle,
            timeStr,
            DoubleToString(rates[i].open, digits),
            DoubleToString(rates[i].high, digits),
            DoubleToString(rates[i].low, digits),
            DoubleToString(rates[i].close, digits),
            IntegerToString(rates[i].tick_volume),
            IntegerToString(rates[i].spread),
            IntegerToString(rates[i].real_volume)
         );
      }
      else if(IncludeSpread)
      {
         FileWrite(handle,
            timeStr,
            DoubleToString(rates[i].open, digits),
            DoubleToString(rates[i].high, digits),
            DoubleToString(rates[i].low, digits),
            DoubleToString(rates[i].close, digits),
            IntegerToString(rates[i].tick_volume),
            IntegerToString(rates[i].spread)
         );
      }
      else if(IncludeRealVolume)
      {
         FileWrite(handle,
            timeStr,
            DoubleToString(rates[i].open, digits),
            DoubleToString(rates[i].high, digits),
            DoubleToString(rates[i].low, digits),
            DoubleToString(rates[i].close, digits),
            IntegerToString(rates[i].tick_volume),
            IntegerToString(rates[i].real_volume)
         );
      }
      else
      {
         FileWrite(handle,
            timeStr,
            DoubleToString(rates[i].open, digits),
            DoubleToString(rates[i].high, digits),
            DoubleToString(rates[i].low, digits),
            DoubleToString(rates[i].close, digits),
            IntegerToString(rates[i].tick_volume)
         );
      }
   }

   FileClose(handle);
   Print("Exported: ", filename, " (", copied, " bars)");

   return true;
}

//+------------------------------------------------------------------+
//| Export comprehensive symbol info to CSV                           |
//+------------------------------------------------------------------+
void ExportSymbolInfo(string symbol, int handle)
{
   // Integer properties
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   int spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   bool spreadFloat = (bool)SymbolInfoInteger(symbol, SYMBOL_SPREAD_FLOAT);
   ENUM_SYMBOL_TRADE_MODE tradeMode = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
   ENUM_SYMBOL_TRADE_EXECUTION tradeExe = (ENUM_SYMBOL_TRADE_EXECUTION)SymbolInfoInteger(symbol, SYMBOL_TRADE_EXEMODE);
   ENUM_SYMBOL_SWAP_MODE swapMode = (ENUM_SYMBOL_SWAP_MODE)SymbolInfoInteger(symbol, SYMBOL_SWAP_MODE);
   ENUM_DAY_OF_WEEK swapRollover3Days = (ENUM_DAY_OF_WEEK)SymbolInfoInteger(symbol, SYMBOL_SWAP_ROLLOVER3DAYS);
   int stopsLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freezeLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   
   // Double properties
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double contractSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double volumeMin = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double volumeMax = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double volumeStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double swapLong = SymbolInfoDouble(symbol, SYMBOL_SWAP_LONG);
   double swapShort = SymbolInfoDouble(symbol, SYMBOL_SWAP_SHORT);
   double marginInitial = SymbolInfoDouble(symbol, SYMBOL_MARGIN_INITIAL);
   double marginMaintenance = SymbolInfoDouble(symbol, SYMBOL_MARGIN_MAINTENANCE);
   double marginHedged = SymbolInfoDouble(symbol, SYMBOL_MARGIN_HEDGED);
   
   // String properties
   string baseCurrency = SymbolInfoString(symbol, SYMBOL_CURRENCY_BASE);
   string quoteCurrency = SymbolInfoString(symbol, SYMBOL_CURRENCY_PROFIT);
   string marginCurrency = SymbolInfoString(symbol, SYMBOL_CURRENCY_MARGIN);
   string description = SymbolInfoString(symbol, SYMBOL_DESCRIPTION);
   string path = SymbolInfoString(symbol, SYMBOL_PATH);
   
   // Calculate derived values
   double spreadPct = (bid > 0) ? (spread * point / bid) * 100 : 0;
   double liveSpread = (ask > 0 && bid > 0) ? (ask - bid) / point : spread;
   
   // Friction score calculation
   double spreadFriction = MathMin(1.0, spreadPct / 0.1);
   double swapFriction = MathMin(1.0, MathMax(MathAbs(swapLong), MathAbs(swapShort)) / 50);
   double frictionScore = (spreadFriction + swapFriction) / 2.0;

   FileWrite(handle,
      symbol,                                    // symbol
      DoubleToString(digits, 0),                 // digits
      DoubleToString(point, 8),                  // point
      DoubleToString(spread, 0),                 // spread_points
      DoubleToString(liveSpread, 1),             // spread_live
      DoubleToString(spreadPct, 4),              // spread_pct
      spreadFloat ? "TRUE" : "FALSE",            // spread_float
      DoubleToString(tickSize, 8),               // tick_size
      DoubleToString(tickValue, 4),              // tick_value
      DoubleToString(contractSize, 2),           // contract_size
      DoubleToString(volumeMin, 4),              // volume_min
      DoubleToString(volumeMax, 2),              // volume_max
      DoubleToString(volumeStep, 4),             // volume_step
      DoubleToString(swapLong, 4),               // swap_long
      DoubleToString(swapShort, 4),              // swap_short
      SwapModeToString(swapMode),                // swap_mode
      DayOfWeekToString(swapRollover3Days),      // swap_rollover3days
      DoubleToString(marginInitial, 4),          // margin_initial
      DoubleToString(marginMaintenance, 4),      // margin_maintenance
      DoubleToString(marginHedged, 4),           // margin_hedged
      DoubleToString(stopsLevel, 0),             // stops_level
      DoubleToString(freezeLevel, 0),            // freeze_level
      TradeModeToString(tradeMode),              // trade_mode
      baseCurrency,                              // base_currency
      quoteCurrency,                             // quote_currency
      marginCurrency,                            // margin_currency
      description,                               // description
      path,                                      // path
      DoubleToString(frictionScore, 4)           // friction_score
   );
}

//+------------------------------------------------------------------+
//| Get error description                                             |
//+------------------------------------------------------------------+
string ErrorDescription(int error)
{
   switch(error)
   {
      case 0:     return "No error";
      case 4001:  return "Unexpected internal error";
      case 4002:  return "Wrong parameter in internal function call";
      case 4003:  return "Wrong parameter when calling system function";
      case 4004:  return "Not enough memory for operation";
      case 4005:  return "Structure contains strings and/or dynamic arrays";
      case 4006:  return "Invalid array size";
      case 4007:  return "Array resize error";
      case 4008:  return "String resize error";
      case 4009:  return "No memory for string";
      case 4010:  return "Wrong string";
      case 4011:  return "Unknown symbol";
      case 4012:  return "Invalid parameter";
      case 4013:  return "Array sorting error";
      case 4014:  return "Wrong trade operation code";
      case 4015:  return "Unknown trade operation";
      case 4401:  return "ERR_CHART_WRONG_ID";
      case 4402:  return "ERR_CHART_NO_REPLY";
      case 5001:  return "Too many files open";
      case 5002:  return "Wrong file name";
      case 5003:  return "Too long file name";
      case 5004:  return "Cannot open file";
      case 5005:  return "Text file buffer error";
      case 5006:  return "Cannot delete file";
      case 5007:  return "Invalid file handle";
      case 5008:  return "Wrong file handle";
      case 5009:  return "File not towrite";
      case 5010:  return "File not toread";
      case 5011:  return "File not bin";
      case 5012:  return "File not txt";
      case 5013:  return "File not txtorcsv";
      case 5014:  return "File not csv";
      case 5015:  return "File readerror";
      case 5016:  return "File binstringerror";
      case 5017:  return "File endoffile";
      case 5018:  return "File notfound";
      case 5019:  return "File cannotrewrite";
      case 5020:  return "Wrong file directory";
      case 5021:  return "File alreadyexists";
      case 5022:  return "Cannot write directory";
      case 5023:  return "Cannot create directory";
      case 5024:  return "Cannot read directory";
      case 5025:  return "File notdirectory";
      default:    return "Unknown error";
   }
}

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("=== KINETRA DATA EXPORT v2.0 ===");
   Print("Export path: ", ExportPath);
   Print("Days: ", DaysToExport);
   Print("Server time: ", TimeToString(TimeCurrent()));
   Print("GMT offset: ", TimeGMTOffset() / 3600, " hours");

   // Create folder
   if(!FolderCreate(ExportPath, FILE_COMMON))
   {
      int error = GetLastError();
      if(error != 5021)  // Folder already exists is OK
      {
         Print("Failed to create folder: ", error);
      }
   }

   // Get all symbols in MarketWatch
   int totalSymbols = SymbolsTotal(true);  // true = only visible in MarketWatch
   Print("Symbols in MarketWatch: ", totalSymbols);

   // Create comprehensive symbol info file if requested
   int infoHandle = INVALID_HANDLE;
   if(ExportSymbolInfo)
   {
      string infoFile = ExportPath + "/symbol_specifications.csv";
      infoHandle = FileOpen(infoFile, FILE_WRITE|FILE_CSV|FILE_COMMON, ',');
      if(infoHandle != INVALID_HANDLE)
      {
         FileWrite(infoHandle, 
            "symbol", "digits", "point", "spread_points", "spread_live", "spread_pct", 
            "spread_float", "tick_size", "tick_value", "contract_size",
            "volume_min", "volume_max", "volume_step",
            "swap_long", "swap_short", "swap_mode", "swap_rollover3days",
            "margin_initial", "margin_maintenance", "margin_hedged",
            "stops_level", "freeze_level", "trade_mode",
            "base_currency", "quote_currency", "margin_currency",
            "description", "path", "friction_score"
         );
      }
   }

   // Build timeframes array
   ENUM_TIMEFRAMES timeframes[];
   int tfCount = 0;

   if(ExportM15) { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_M15; }
   if(ExportM30) { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_M30; }
   if(ExportH1)  { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_H1; }
   if(ExportH4)  { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_H4; }
   if(ExportD1)  { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_D1; }

   int exportedCount = 0;
   int failedCount = 0;

   // Export each symbol
   for(int i = 0; i < totalSymbols; i++)
   {
      string symbol = SymbolName(i, true);
      Print("Processing: ", symbol, " (", i+1, "/", totalSymbols, ")");

      // Export symbol info
      if(infoHandle != INVALID_HANDLE)
         ExportSymbolInfo(symbol, infoHandle);

      // Export each timeframe
      for(int t = 0; t < tfCount; t++)
      {
         if(ExportSymbolData(symbol, timeframes[t], ExportPath))
            exportedCount++;
         else
            failedCount++;
      }
   }

   if(infoHandle != INVALID_HANDLE)
      FileClose(infoHandle);

   Print("=== EXPORT COMPLETE ===");
   Print("Exported: ", exportedCount, " files");
   Print("Failed: ", failedCount, " files");
   Print("Location: MQL5/Files/", ExportPath, " (Common folder)");

   // Show completion message
   string message = StringFormat(
      "Export complete!\n\n"
      "Files exported: %d\n"
      "Files failed: %d\n"
      "Location: MQL5/Files/%s\n\n"
      "Copy to: ~/Kinetra/data/",
      exportedCount, failedCount, ExportPath
   );
   
   MessageBox(message, "Kinetra Data Export", MB_ICONINFORMATION);
}
//+------------------------------------------------------------------+
