# Federated Learning with Flower library

This repository demonstrates how to use the [Flower](https://flower.dev/) library to implement federated learning in Python. Federated learning is a technique that allows multiple clients to collaboratively train a machine learning model without sharing their data with a central server.

## Requirements

You can install the required packages using `pip install -r requirements.txt`.

## Data

The data used in this example is a CSV file containing information about different types of Medical test report results, such as HAEMOGLOBINS. The data is split into two parts: one for client1 and one for client2. Each client has access to only their own data.

## Model

The model used in this example is a simple Sequential Model that predicts if the patient should be admitted in the hospital or not. The model is initialized on the server and then distributed to the clients for training.

## How to run

To run this example, you need to follow these steps:

1. Clone this repository using `git clone https://github.com/ZeeshanGeoPk/federated-leaning-with-flower-library.git`.
2. Navigate to the repository folder using `cd federated-leaning-with-flower-library`.
3. Start the server using `python3 server.py`.
4. In another terminal, start client1 using `python3 client1.py`.
5. In another terminal, start client2 using `python3 client2.py`.
6. Wait for the training to finish and observe the results on the server terminal.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ZeeshanGeoPk/federated-leaning-with-flower-library/blob/main/LICENSE) file for details.
