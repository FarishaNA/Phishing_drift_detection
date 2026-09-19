PhishDrift - phishing email classifier with drift monitoring
=============================================================

1. Unzip this whole folder somewhere (for example Desktop\Farisha\phish_drift_ui).
   Do NOT copy files out of it one by one.
2. Put the notebook outputs in the artifacts folder (see artifacts\PUT_FESAD_OUTPUTS_HERE.txt).
3. One time only:   pip install scikit-learn==1.6.1
                    pip install pandas numpy scipy matplotlib joblib beautifulsoup4
4. Double-click run.bat   (or in a terminal inside this folder:  python server.py)
   Your browser opens http://localhost:8000 . Press Ctrl+C in the black window to stop.

There is no Streamlit in this version. Do not use "streamlit run".
Files:  server.py (web server)   charts.py (charts)   pipeline.py (email cleaning + models)
        web\style.css, web\main.js, web\logo.svg (the page design)   export_inbox.py (inbox -> CSV helper)
