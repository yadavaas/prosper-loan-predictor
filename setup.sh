mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
[theme]\n\
base=\"light\"\n\
secondaryBackgroundColor=\"#328e8e\"\n\
textColor=\"#433939\"\n\
" > ~/.streamlit/config.toml
