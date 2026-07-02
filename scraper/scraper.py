from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Setup Chrome
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Open SHL Product Catalog
driver.get("https://www.shl.com/solutions/products/product-catalog/")

# Wait for page to load
time.sleep(5)

# Get page source
html = driver.page_source

driver.quit()

# Parse HTML
soup = BeautifulSoup(html, "html.parser")

# Save HTML for inspection
with open("catalog.html", "w", encoding="utf-8") as f:
    f.write(soup.prettify())

print("HTML saved as catalog.html")