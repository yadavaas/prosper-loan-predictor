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
primaryColor = "\#70f9b3"\n\
backgroundColor = "\#0b2d4c"\n\
secondaryBackgroundColor = "\#4c4747"\n\
textColor = "\#ffffff"\n\
" > ~/.streamlit/config.toml
