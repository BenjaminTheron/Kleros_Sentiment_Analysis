# Imports
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

# Flag to determine if the entire program should run, or only the number of cases needs
# to be collected
EXECUTION_FLAG = False

# Initialising the webdriver object
driver_obj = webdriver.Chrome()

# Opening the kleros dashboard containing all the disputes
driver_obj.get("https://klerosboard.com/1/cases")

# The implicit wait time for the driver is set to 15 seconds
driver_obj.implicitly_wait(10)

############### Interacting with the webpage ###############

# Getting the webpage element that contains the rows for each case
cases = driver_obj.find_element(By.CLASS_NAME, "MuiDataGrid-virtualScrollerRenderZone.css-1inm7gi")

# Storing the selenium object for each case in a list
individual_cases = cases.find_elements(By.CLASS_NAME, "MuiDataGrid-cell--withRenderer.MuiDataGrid-cell.MuiDataGrid-cell--textRight")

# Creating the outline for the JSON object to be written to the data file
data = dict()

# Find the total number of cases -> gets the number of the most recent case\
# (necessary to access the page containing the case data for each court)
NUM_CASES = int(individual_cases[0].text[1::])

if EXECUTION_FLAG != False:
    # TEMPORARY TERMINAL END POINT FOR THE FOR LOOP
    temp_end = 1420

    # Extracting the data for each case by navigating to the relevant URL
    for i in range(NUM_CASES, -1, -1):
        print(f"caseID:{i}")
        # Initialises the dictionaries used to store the votes, evidence and outcomes for each round in a case
        vote_list = dict()
        case_evidence = dict()
        outcomes = dict()

        # Navigate to the URL for the current case
        current_case = f"https://klerosboard.com/1/cases/{i}"
        driver_obj.get(current_case)

        try:
            end_time = time.time() + 30
            while time.time() < end_time:
                # Until the case content has loaded try refreshing the page every
                # 5 seconds (up to a 30s wait)
                time.sleep(5)
                try:
                    # Attempt to find the page data
                    case_data = driver_obj.find_element(By.CLASS_NAME,
                                                    "MuiGrid-root.MuiGrid-container.MuiGrid-spacing-xs-2.css-1eak6tf")
                    # If it doesn't throw an exception (the page data is found) return to the main program flow
                    # and extract the case data
                    break
                except:
                    # If the case data isn't found, refresh the page and wait 5 seconds
                    driver_obj.refresh()

            partial_data = driver_obj.find_element(By.CLASS_NAME,
                                                "MuiGrid-root.MuiGrid-container.MuiGrid-spacing-xs-2.css-1eak6tf")
            
            # Extract which court the case is occuring in
            court = partial_data.find_elements(By.CLASS_NAME,
                                            "MuiTypography-root.MuiTypography-inherit.MuiLink-root.MuiLink-underlineAlways.css-1h5vh24")
            court = court[0].text

            # Extract the date and appeal data
            date_appeal = partial_data.find_elements(By.CLASS_NAME,
                                                    "MuiGrid-root.MuiGrid-item.css-15y1fg1")
            
            # Using string slicing to extract only the date from date_appeal
            date_text = date_appeal[0].text
            date = date_text.split('\n')
            date = date[-1].split(',')
            date = date[0]

            # Using string slicing to get the number of rounds in the current case
            rounds_text = date_appeal[-1].text
            rounds = rounds_text.split('\n')
            rounds = rounds[-1]

            # Finds the selenium object linking to the 'box' holding the results for each round
            results_section = driver_obj.find_element(By.CLASS_NAME, "MuiBox-root.css-vcpz9t")
            # Buttons indicating the round are index 0
            # Information for each round are index 1

            # Data not in the currently selected round is 'hidden' and cannot be accessed unless clicked
            for j in range(0, int(rounds)):
                round_data = results_section.find_elements(By.CLASS_NAME, "MuiGrid-root.MuiGrid-container.css-1mq3rge")
                # Extract the outcome of the current round
                outcome = round_data[0].find_elements(By.CLASS_NAME, "MuiTypography-root.MuiTypography-body1.css-15yejx0")
                # Extract the votes for the current round -> stored as a dict to maintain JSON formatting
                votes_dict = dict()
                current_votes = results_section.find_elements(By.CLASS_NAME,
                                                    "MuiPaper-root.MuiPaper-elevation.MuiPaper-"+\
                                                    "rounded.MuiPaper-elevation1.MuiAccordion-root"+\
                                                    ".MuiAccordion-rounded.MuiAccordion-gutters.css-18poalt")
                
                for k in range(0, len(current_votes)):
                    votes_dict[k] = current_votes[k].find_element(By.CLASS_NAME, "MuiTypography-root.MuiTypography-body1.css-1f3iloc").text
                
                # Store the data for the current round, maintaining JSON format, where the rounds are stored most->least recent.
                outcomes[int(rounds) - j - 1] = outcome[2].text
                vote_list[int(rounds) - j - 1] = votes_dict
                # Click on the next button for the next round to extract the data from it
                if j != (int(rounds) - 1):
                    results_section.find_element(By.ID, f"Tab-{int(rounds)-2-j}").click()

            ############### Validation required to extract the evidence ###############
            # Navigate to the case details page for the current case
            driver_obj.get(f"https://court.kleros.io/cases/{i}")

            # KNOWN ISSUE -> The evidence page for some cases doesn't load
            # To counteract this, the page is given 10 seconds to load (as given by globally
            # setting the implicit wait to 10 seconds at the start of the program), if it hasn't,
            # nothing is stored for the evidence
            try:
                # For testing purposes/ collecting missed evidence 
                # MAKES EXECUTION DRASTICALLY SLOWER
                time.sleep(10)
                evidence = driver_obj.find_element(By.CLASS_NAME, "sc-gEvEer.hmaUxN")

                # Waiting until the evidence tab is clickable
                evidence_clickable = WebDriverWait(driver_obj, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "sc-eqUAAy.dAeLDz"))
                )

                # Using JavaScript to bring the element in view
                driver_obj.execute_script("arguments[0].scrollIntoView(true);", evidence)

                # Then using the actions class to click the evidence tab
                actions = ActionChains(driver_obj)
                actions.move_to_element(driver_obj.find_element(By.CLASS_NAME, "sc-eqUAAy.dAeLDz")).click().perform()

                # Gets each piece of evidence submitted by the jurors
                each_evidence = evidence.find_elements(By.CLASS_NAME, "ant-card.sc-fqkvVR.fALeOq.ant-card-bordered")

                # Extracts and stores the title and description for each piece of evidence
                for j in range(0, len(each_evidence)):
                    case_evidence[j] = {"title": each_evidence[j].find_element(By.CLASS_NAME, "sc-dcJsrY.hdfsbl").text,
                                        "description": each_evidence[j].find_element(By.CLASS_NAME, "sc-iGgWBj.cFKPim").text}
                    
                # Extract the description of the court case
                description = driver_obj.find_element(By.CLASS_NAME,
                                                    "ant-card.case-details-card__StyledInnerCard-sc-1cab6gv-13.cQfYkO.ant-card-bordered")
                desc = description.find_elements(By.CLASS_NAME, "ant-card-body")
                desc = desc[0].text
            except:
                # In the event that the page doesn't load within 10 seconds
                # No evidence is stored for the case
                print(f"Evidence for case {i} did not load!")

            # VALIDATION STEP
            print(case_evidence)

            # Adds all the case data to the JSON data object
            data[i] = {
                "appeal": True if int(rounds) > 1 else False,
                "description": desc,
                "evidence": case_evidence,
                "outcome": outcomes,
                "court": court,
                "votes": vote_list,
                "start_date": date
            }

            # Every 200 cases, write the data to the results file and reset the data dictionary
            # TODO CHANGE BACK CURRENTLY STORING EVERY 50 CASES
            if (((NUM_CASES - i) % 50 == 0) or (i == temp_end)) and (i < NUM_CASES):
                # Writes the json data to a file
                with open("data.json", "a") as file:
                    json.dump(data, file, indent=4)

                # Take a 5 second break
                time.sleep(5)

                # Reset the data dictionary
                data = dict()

        except:
            print(f"Case data for ID {i} did not load in time.")
            # The data stored for this case is a collection of empty sets
            data[i] = {
                "appeal": {},
                "description": {},
                "evidence": {},
                "outcome": {},
                "court": {},
                "votes": {},
                "start_date": {}
            }

print("End of program...")

# Closing the browser
driver_obj.quit()