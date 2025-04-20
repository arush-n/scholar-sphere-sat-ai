from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from q_parser import q_parse_no_images
import time
import json
import os
import re

def webscraper(genre, domain, topic, exclude_active):
    genre_num = 2 if genre == 'Reading' else 3
    # Set up Selenium
    path = "C:\\Users\\rapto\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe" # put path to chromedriver here
    # Load the dynamic webpage
    url = 'https://satsuitequestionbank.collegeboard.org'

    service = Service(executable_path=path)
    driver = webdriver.Chrome(service=service)
    driver.get(url)

    qN = 1
    all_questions = []
    # Wait for the page to fully load
    time.sleep(5)
    driver.find_element(By.XPATH, '//*[@id="home-banner"]/div[2]/button').click()
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="selectAssessmentType"]').click()
    time.sleep(.5)
    driver.find_element(By.XPATH, '//*[@id="selectAssessmentType"]/option[2]').click()
    time.sleep(1.5)
    driver.find_element(By.XPATH, '//*[@id="selectTestType"]').click()
    time.sleep(.5)
    driver.find_element(By.XPATH, '//*[@id="selectTestType"]/option['+str(genre_num)+']').click() 
    time.sleep(3)
    driver.find_element(By.XPATH, list(domain.values())[0]).click()
    time.sleep(.5)
    driver.find_element(By.XPATH, '//*[@id="search"]/div/div[2]/div/button[2]').click()
    time.sleep(1.5)
    if exclude_active:
        driver.find_element(By.XPATH, '//*[@id="apricot_check_4"]').click()
    time.sleep(.1)
    driver.find_element(By.XPATH, '//*[@id="dropdown2"]').click()
    time.sleep(.5)
    driver.find_element(By.XPATH, str(topic)).click()
    time.sleep(1.5)
    
    # Locate and click the specific question modal (you can customize this step)
    try:
        while True:  
            # Click the button in each row
            time.sleep(1.5)
            button_xpath = f'/html/body/div[2]/div[3]/div[2]/div/div[3]/div[2]/div/div[2]/div/div[2]/table/tbody/tr[{qN}]/td[2]/button'
            
            button = driver.find_element(By.XPATH, button_xpath)
            button.click()
            
            # Wait for the modal to load (adjust the time as necessary)
            time.sleep(1.5)
            
            # Find the dynamic modal content
            modal_content = driver.find_element(By.XPATH, '//*[@class="question-info"]')
            # Parse the modal content using BeautifulSoup
            html_content = modal_content.get_attribute('outerHTML')

            # Parse the HTML content and extract information using q_parser
            question_info = q_parse_no_images(html_content, genre)
                
            # Add the question to the list
            all_questions.append(question_info)
            
            # Close the modal (if necessary) or navigate back   
            close_button = driver.find_element(By.XPATH, '//*[@id="modalID1"]/div/div/div/div/div/div[1]/button')
            close_button.click()
            
            # Wait for the modal to close
            time.sleep(1)
            
            # Increment qN to move to the next row
            qN += 1
    except Exception as e:
        print('error', e)

    finally:
        # Convert the collected data to JSON format
        parsed_data = json.dumps(all_questions, indent=4)

        # Replace spaces with hyphens in the topic variable to create a valid filename
        topic_filename = re.sub(r'[^a-zA-Z0-9]', '-', re.search(r'"([^"]+)"', topic).group(1)).lower() + ".json"


        # Create the directory structure: question_data/genre/domain/
        folder_path = os.path.join("question_data", genre, list(domain.keys())[0])

        # Ensure the directories exist
        os.makedirs(folder_path, exist_ok=True)

        # Create the full file path with the topic as the filename
        file_path = os.path.join(folder_path, topic_filename)

        # Write the parsed data to the JSON file
        with open(file_path, "w") as json_file:
            json_file.write(parsed_data)

        # Close the browser when done
        driver.quit()