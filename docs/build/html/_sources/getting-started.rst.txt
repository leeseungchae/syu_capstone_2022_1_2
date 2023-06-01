Getting Started
==============
This guide will walk you through the steps to get started with BloomingMind.

Installation
------------

1. Clone the BloomingMind repository:

.. code-block:: bash

    git clone https://github.com/leeseungchae/syu_capstone_2022_1_2.git

2. Navigate to the project directory:

.. code-block:: bash

    cd BloomingMind

3. create and activate a virtual environment (optional but recommended):

.. code-block:: bash

    python3 -m venv env
    source env/bin/activate

4. Install the required dependencies:

.. code-block:: bash

    pip install -r requirements.txt
Usage
-----

To run the BloomingMind application, follow these steps:

.. code-block:: python

   1. Open your terminal and navigate to the `BloomingMind` directory:
      cd BloomingMind

   2. Use the following command to start the application:
      gunicorn BloomingMind.asgi.dev:application -b 0.0.0.0:8000 -w 1 -k uvicorn.workers.UvicornWorker --reload

Configuration
-------------

MyProject supports various configuration options. You can configure it by modifying the `myproject.conf` file.

