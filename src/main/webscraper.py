from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import json
import q_parser


x  = 'Math'
genre = 2 if x == 'Reading' else 3
math_domain = {'Algebra': '//*[@id="checkbox-algebra"]', 'Advanced Math': '//*[@id="checkbox-advanced math"]', 'Problem Solving and Data Analysis': '//*[@id="checkbox-problem-solving and data analysis"]','Geometry and Trigonometry': '//*[@id="checkbox-geometry and trigonometry"]'}
topics = ['Linear functions', 'Quadratic equations', 'Exponential functions', 'Systems of equations']
exclude_active = False

def webscraper(genre, math_domain, topics, exclude_active):
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
    driver.find_element(By.XPATH, '//*[@id="selectTestType"]/option['+str(genre)+']').click() 
    time.sleep(3)
    driver.find_element(By.XPATH, list(math_domain.values())[0]).click()
    time.sleep(.5)
    driver.find_element(By.XPATH, '//*[@id="search"]/div/div[2]/div/button[2]').click()
    time.sleep(1.5)
    if exclude_active:
        driver.find_element(By.XPATH, '//*[@id="apricot_check_4"]').click()
    time.sleep(.1)

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
            soup = BeautifulSoup(html_content, 'html.parser')
            # Extract the question content
            question = {
                "assessment": soup.find('div', string='SAT').text if soup.find('div', string='SAT') else 'N/A',
                "test": soup.find('div', string=genre).text if soup.find('div', string=genre) else 'N/A',
                #"domain": soup.find('div', string=list(math_domain.keys())[0]).text if soup.find('div', string=list(math_domain.keys())[0]) else 'N/A',
                "skill": next((soup.find('div', string=topic).text for topic in topics if soup.find('div', string=topic)), 'N/A'),
                "difficulty": soup.find('span', class_='tqdifficulty')['aria-label'] if soup.find('span', class_='tqdifficulty') else 'N/A',
                "question_id": soup.find('h5', class_='question-id').text.split(": ")[1] if soup.find('h5', class_='question-id') else 'N/A',
                "question_content": soup.find('div', class_='question') if soup.find('div', class_='question') else 'N/A',
                "answer_choices": soup.find_all('li') if soup.find_all('li') else 'N/A',
                "correct_answer": soup.find('div', class_='correct-answer').find('p').text if soup.find('div', class_='correct-answer') and soup.find('div', class_='correct-answer').find('p') else 'N/A',
                "rationale": soup.find('div', class_='rationale') if soup.find('div', class_='rationale') else 'N/A',
                "difficulty": soup.find('div', class_='question-difficulty').find('p').text.strip() if soup.find('div', class_='question-difficulty') and soup.find('div', class_='question-difficulty').find('p') else 'N/A'
            }
            
            # Adding logic to detect image links and append them to the respective content
            def extract_content_with_images(element):
                content = []
                for child in element.find_all(recursive=True):
                    if child.name == 'img' and child.get('src', '').startswith('data'):
                        content.append(f'[Image: {child["src"]}]')
                    else:
                        content.append(child.get('alttext', child.text))
                return " ".join(content).strip()

            if soup.find('div', class_='question'):
                question["question_content"] = extract_content_with_images(soup.find('div', class_='question'))

            if soup.find('div', class_='rationale'):
                question["rationale"] = extract_content_with_images(soup.find('div', class_='rationale'))
                
            # Add the question to the list
            all_questions.append(question)
            
            # Close the modal (if necessary) or navigate back   
            close_button = driver.find_element(By.XPATH, '//*[@id="modalID1"]/div/div/div/div/div/div[1]/button')
            close_button.click()
            
            # Wait for the modal to close
            time.sleep(1)
            
            # Increment qN to move to the next row
            qN += 1
            
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Convert the collected data to JSON format
        parsed_data = json.dumps(all_questions, indent=4)

        # Output the parsed JSON data
        #print(parsed_data)

        # Close the browser when done
        driver.quit()