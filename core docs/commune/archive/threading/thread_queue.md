# Readme for Code "ThreadQueue and ProducerThread"

This Python code provides utility classes for managing a queue with the assistance of producer and threading capabilities. This code was created by Yuma Rao under the MIT License.

## Class Breakdown

### 1. ProducerThread

This class is a custom threading class that works in the background to fill a specified queue with the results of a target function. In its initializer, it takes a target function, along with optional arguments for that function, and a queue that it should fill. It utilizes the `run` method to continuously check if the queue is full and run the target function to fill the queue if it isn't. Additionally, it has `stop` and `stopped` methods to control the execution of the thread.

### 2. ThreadQueue

This class manages the queue and the associated ProducerThread that continuously monitors and fills the queue. It takes a maximum number of threads as an argument. On instantiation, it sets up the queue, creates a producer thread instance, and starts the thread. It also includes a `close` method to stop the producer thread and join it back to the main thread. This method is also invoked when an instance of the class is being deleted.

## Usage

To use this code, you should first define a target function and a queue that the function will fill. This function is then passed to the ProducerThread along with any arguments it needs. Then, the ThreadQueue class is instantiated with the producer_target and producer_arg being passed in as arguments.

Finally, upon successful execution of the queue, the Processing thread can be stopped and, if necessary, joined back to the current main thread.

## Requirements

This code requires Python 3. The Python code also utilizes the following modules: threading, time, queue, and loguru. The 'loguru' module is not a part of the Standard Python Library, therefore you may need to install it manually. Also, ensure that your Python version supports multi-threading capabilities.