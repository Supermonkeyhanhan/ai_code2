# PDF viewer fix

The app no longer calls `st.pdf()`, so it does not require the optional `streamlit-pdf` component.
The supplied `assets/campus_map.pdf` is displayed with a browser-native base64 PDF iframe and can also be downloaded.

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```
