from locust import HttpUser, TaskSet, task, between
from data_stresst_test import  data_test

class MyTaskSet(TaskSet):
    @task
    def my_task(self):
        # Define the API endpoint and request payload
        endpoint = "/phase-2/prob-1/predict"
        payload = data_test

        # Make a POST request to the API endpoint
        response = self.client.post(endpoint, json=payload)

        # Validate the response if needed
        if response.status_code == 200:
            # Perform validation or assertions on the response

            # Example: Retrieve and print the response data
            response_data = response.json()
            # print(response_data)

    @task
    def my_task2(self):
        # Define the API endpoint and request payload
        endpoint = "/phase-2/prob-2/predict"
        payload = data_test

        # Make a POST request to the API endpoint
        response = self.client.post(endpoint, json=payload)

        # Validate the response if needed
        if response.status_code == 200:
            # Perform validation or assertions on the response

            # Example: Retrieve and print the response data
            response_data = response.json()
            # print(response_data)


class MyUser(HttpUser):
    tasks = [MyTaskSet]
    wait_time = between(1, 3)  # Wait time between consecutive requests

    # Optional: Add setup and teardown methods
    def on_start(self):
        # Perform any setup actions before the test starts
        pass

    def on_stop(self):
        # Perform any cleanup actions after the test ends
        pass
