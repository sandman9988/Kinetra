//+------------------------------------------------------------------+
//|                                              ExportMarketData.mq5 |
//|                                                          Kinetra |
//|                         Exports MarketWatch data to CSV files    |
//+------------------------------------------------------------------+
#property copyright "Kinetra"
#property link      ""
#property version   "1.00"
#property script_show_inputs

//--- Input parameters
input int      DaysToExport = 365;           // Days of history
input string   ExportPath = "KinetraData";   // Export folder name (in MQL5/Files/)
input bool     ExportM15 = true;             // Export M15
input bool     ExportM30 = true;             // Export M30
input bool     ExportH1 = true;              // Export H1
input bool     ExportH4 = true;              // Export H4

//+------------------------------------------------------------------+
//| Get timeframe as string                                           |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
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

   // Get rates
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(symbol, timeframe, startDate, endDate, rates);

   if(copied <= 0)
   {
      Print("No data for ", symbol, " ", TimeframeToString(timeframe));
      return false;
   }

   // Generate filename
   string tfStr = TimeframeToString(timeframe);
   string startStr = TimeToString(rates[copied-1].time, TIME_DATE);
   string endStr = TimeToString(rates[0].time, TIME_DATE);
   StringReplace(startStr, ".", "");
   StringReplace(endStr, ".", "");

   string filename = folder + "/" + symbol + "_" + tfStr + ".csv";

   // Open file
   int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("Failed to create file: ", filename);
      return false;
   }

   // Write header
   FileWrite(handle, "time", "open", "high", "low", "close", "volume", "spread");

   // Write data (oldest first)
   for(int i = copied - 1; i >= 0; i--)
   {
      FileWrite(handle,
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
         DoubleToString(rates[i].open, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         DoubleToString(rates[i].high, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         DoubleToString(rates[i].low, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         DoubleToString(rates[i].close, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         IntegerToString(rates[i].tick_volume),
         IntegerToString(rates[i].spread)
      );
   }

   FileClose(handle);
   Print("Exported: ", filename, " (", copied, " bars)");

   return true;
}

//+------------------------------------------------------------------+
//| Export symbol info to JSON                                        |
//+------------------------------------------------------------------+
void ExportSymbolInfo(string symbol, int handle)
{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD) * point;
   double spreadPct = (spread / SymbolInfoDouble(symbol, SYMBOL_BID)) * 100;
   double contractSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double swapLong = SymbolInfoDouble(symbol, SYMBOL_SWAP_LONG);
   double swapShort = SymbolInfoDouble(symbol, SYMBOL_SWAP_SHORT);
   double volumeMin = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double volumeMax = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);

   // Calculate friction score
   double spreadFriction = MathMin(1.0, spreadPct / 0.1);
   double swapFriction = MathMin(1.0, MathMax(MathAbs(swapLong), MathAbs(swapShort)) / 50);
   double frictionScore = 0.7 * spreadFriction + 0.3 * swapFriction;

   FileWrite(handle,
      symbol,
      DoubleToString(digits, 0),
      DoubleToString(point, 8),
      DoubleToString(SymbolInfoInteger(symbol, SYMBOL_SPREAD), 0),
      DoubleToString(spreadPct, 4),
      DoubleToString(contractSize, 2),
      DoubleToString(swapLong, 2),
      DoubleToString(swapShort, 2),
      DoubleToString(volumeMin, 2),
      DoubleToString(volumeMax, 2),
      DoubleToString(frictionScore, 4)
   );
}

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("=== KINETRA DATA EXPORT ===");
   Print("Export path: ", ExportPath);
   Print("Days: ", DaysToExport);

   // Create folder
   FolderCreate(ExportPath, FILE_COMMON);

   // Get all symbols in MarketWatch
   int totalSymbols = SymbolsTotal(true);  // true = only visible in MarketWatch
   Print("Symbols in MarketWatch: ", totalSymbols);

   // Create symbol info file
   string infoFile = ExportPath + "/symbol_info.csv";
   int infoHandle = FileOpen(infoFile, FILE_WRITE|FILE_CSV|FILE_COMMON, ',');
   if(infoHandle != INVALID_HANDLE)
   {
      FileWrite(infoHandle, "symbol", "digits", "point", "spread", "spread_pct",
                "contract_size", "swap_long", "swap_short", "vol_min", "vol_max", "friction");
   }

   // Timeframes to export
   ENUM_TIMEFRAMES timeframes[];
   int tfCount = 0;

   if(ExportM15) { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_M15; }
   if(ExportM30) { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_M30; }
   if(ExportH1)  { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_H1; }
   if(ExportH4)  { ArrayResize(timeframes, tfCount+1); timeframes[tfCount++] = PERIOD_H4; }

   int exportedCount = 0;

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
      }
   }

   if(infoHandle != INVALID_HANDLE)
      FileClose(infoHandle);

   Print("=== EXPORT COMPLETE ===");
   Print("Exported ", exportedCount, " files");
   Print("Location: MQL5/Files/", ExportPath);

   // Show message
   MessageBox("Export complete!\n\n" +
              "Files: " + IntegerToString(exportedCount) + "\n" +
              "Location: MQL5/Files/" + ExportPath + "\n\n" +
              "Copy to ~/Kinetra/data/master/",
              "Kinetra Data Export", MB_ICONINFORMATION);
}
//+------------------------------------------------------------------+
