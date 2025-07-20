import ollama
import sys

def check_llama_installation():
    print("Checking Llama installation...")
    
    # 1. Check Python package
    try:
        import ollama
        print("✅ Ollama Python package is installed")
    except ImportError:
        print("❌ Ollama Python package not found")
        print("Please install with: pip install ollama")
        return False
    
    # 2. Check Ollama service
    try:
        client = ollama.Client()
        print("✅ Ollama service is running")
    except ConnectionError:
        print("❌ Could not connect to Ollama service")
        print("Please start Ollama: \n  Linux/macOS: ollama serve \n  Windows: run the Ollama app")
        return False
    
    # 3. Check model availability
    try:
        models = ollama.list()['models']
        model_names = [model['name'] for model in models]
        
        print(f"Available models: {', '.join(model_names)}")
        
        if any('llama3' in name for name in model_names):
            print("✅ llama3 model is available")
            return True
        else:
            print("❌ llama3 model not found")
            print("Download with: ollama pull llama3")
            return False
            
    except Exception as e:
        print(f"❌ Model check failed: {str(e)}")
        return False

if __name__ == "__main__":
    if check_llama_installation():
        print("\nLlama is ready to use!")
        print("Testing with a simple prompt...")
        
        try:
            response = ollama.generate(
                model='llama3',
                prompt='What is 2+2? Respond with just the number.',
                options={'temperature': 0.0}
            )
            print(f"Response: {response['response']}")
            print("✅ Llama is working correctly")
        except Exception as e:
            print(f"❌ Llama test failed: {str(e)}")
    else:
        print("\nLlama setup is incomplete")
        sys.exit(1)