

# Setup Instructions

## 1. Install `pyenv`

If you don't have `pyenv` installed, you can install it using the following commands:

```bash
curl https://pyenv.run | bash
```


## 4. Create and Activate VirtualEnv
Create a virtual environment using pyenv:
```bash
pyenv virtualenv 3.9.7 myenv
```
Activate the virtual environment:
```bash
pyenv activate myenv
```


## 5. Set Up API Keys
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 6. Set Up API Keys
If your application requires API keys, create a .env file in the root directory of your project and add your API keys there. For example:
```bash
API_KEY=your_api_key_here
ANOTHER_API_KEY=another_api_key_here
```

## 7. Run your application:
```bash
streamlit run app.py
```
