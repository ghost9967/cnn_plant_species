#code lang utf-8
#@author Arpan Sarkar
import requests
import sqlite3
import time
import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
#v0.01 discarded due to dependencies on CSS selectors __commented out__
#v0.02
def connect_page_old():
    page = requests.get("https://agmarknet.gov.in")
    print("Status: ",page.status_code)
    if(page.status_code==200):
        print("Successfully Fetched from https://agmarknet.gov.in")
        return page
    else:
        print("Page Not Loaded")

def load():
    df = pd.read_csv("datafile.csv")
    print("Load Successfull")
    df_ref = df[['state','district','commodity','min_price','modal_price']]
    print(df_ref.head())
    return df_ref
    
def connect_db():
    conn = sqlite3.connect("prices.db")
    if(conn):
        print("Connected Successfully")
        return conn
    else:
        print("Connection Failed")
        
def fetch_price():
    username = os.getenv("USERNAME")
    userProfile = "C:\\Users\\" + username + "\\AppData\\Local\\Google\\Chrome\\User Data\\Default"
    options = webdriver.ChromeOptions()
    options.add_argument("user-data-dir={}".format(userProfile))
    options.binary_location = ""
    driver = webdriver.Chrome('C:/Users/sarka/Desktop/Major Project/Latest/Input/Code/chromedriver_win32/chromedriver',options=options)
    driver.get('https://data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi')
    #click download csv
    time.sleep(15)
    print("Solve Captcha")
    dl_button = driver.find_elements_by_xpath("//a[@title='csv' and @class='csv']")[0]
    dl_button.click()
    time.sleep(5)
    c = driver.find_element_by_id("edit-download-reasons-2")
    ActionChains(driver).move_to_element(c).click(c).perform()  
    driver.execute_script("document.getElementById('edit-reasons-d-3').click()");
    #d = driver.find_element_by_id("edit-reasons-d-3")
    #d.click()
    #ActionChains(driver).move_to_element(d).click(d).perform()
    sub = driver.find_elements_by_xpath("//input[@name='op' and @value='Submit']")[0]
    sub.click()
    time.sleep(15)
    driver.quit()
    print("File downloaded in Default Download Location")

def create_price_db():
    #page = connect_page_old() v0.01
    #soup = BeautifulSoup(page.content, 'html.parser')
    #soup.prettify()
    #cereals = soup.find(id="gvCustomers_gvOrders_0")
    conn = connect_db()
    df = load()
    sql = "CREATE TABLE IF NOT EXISTS price (ID INT NOT NULL, LOCATION VARCHAR(20) NOT NULL, COMODITY VARCHAR(20) NOT NULL, DISTRICT VARCHAR(10) NOT NULL, MIN_PRICE VARCHAR(8), MODAL_PRICE VARCHAR(8), PRIMARY KEY(ID))"
    conn.execute(sql)
    state = df['state']
    district = df['district']
    commodity = df['commodity']
    min_price = df['min_price']
    modal_price = df['modal_price']
    for i in range(len(state)):
        sql1 = "INSERT INTO price VALUES ("+str(i)+",'"+str(state[i])+"','"+str(commodity[i])+"','"+str(district[i])+"',"+str(min_price[i])+","+str(modal_price[i])+")"
        print(sql1)
        conn.execute(sql1)
        time.sleep(0.12)
        print(i+1 , " Records Inserted")
    conn.commit()
    conn.close()

def view_price_db():
    rows = fetch_db()
    for x in rows:
        print(x," | ")

def update_price_db():
    conn = connect_db()
    df = load()
    state = df['state']
    district = df['district']
    commodity = df['commodity']
    min_price = df['min_price']
    modal_price = df['modal_price']
    for i in range(len(state)):
        sql = sql = "UPDATE price SET LOCATION = '"+str(state[i])+"',COMODITY = '"+str(commodity[i])+"', DISTRICT = '"+str(district[i])+"', MIN_PRICE = "+str(min_price[i])+", MODAL_PRICE = "+str(modal_price[i])+" WHERE ID = "+str(i)+";"
        print(sql)
        conn.execute(sql)
        time.sleep(0.12)
        print(i+1 , " Records Updated")
    conn.commit()
    conn.close()

def fetch_db():
    conn = connect_db()
    cur = conn.cursor()
    sql = "SELECT * FROM price"
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return rows

def view_species():
    conn = connect_db()
    cur = conn.cursor()
    sql = "SELECT DISTINCT(COMODITY) FROM price"
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    i=1
    for x in rows:
        print(i," ",x)
        i+=1

#main start
print("\t\tMENU\t\t")
print("\t    PRICE SCRAPER\t\t")
print("\tDeveloped By Arpan Sarkar\t\t")
print(" ")
while(1):
    print("1. Scrape Price Data [Internet Connection Needed]")
    print("2. Create Price Database [First Time Only]")
    print("3. Update Price Database [Prefetch Price Data]")
    print("4. View Price Database")
    print("5. View Commodities in Database")
    print("6. Exit")
    choice = input("Enter your choice")
    if choice=='1':
        fetch_price()
    elif choice=='2':
        create_price_db()
    elif choice=='3':
        update_price_db()
    elif choice=='4':
        view_price_db()
    elif choice=='5':
        view_species()
    elif choice=='6':
        break
    else:
        print("Invalid Choice! Try Again")
            
