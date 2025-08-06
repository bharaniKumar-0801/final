import ollama
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.active_model = None
        return cls._instance

    def initialize_qwen_model(self, model_name="qwen2.5:1.5b"):
        try:
            # Ensure the model is pulled
            ollama.pull(model_name)
            return model_name
        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {str(e)}")
            return None

    def initialize_llama_model(self, model_name="llama2:7b"):
        try:
            # Ensure the model is pulled
            ollama.pull(model_name)
            return model_name
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {str(e)}")
            return None
    
    def initialize_deepseek_coder_model(self, model_name="deepseek-coder:1.3b"):
        try:
            # Ensure the model is pulled
            ollama.pull(model_name)
            return model_name
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek Coder model: {str(e)}")
            return None
    
    def initialize_phi_model(self, model_name="phi3.5:latest"):
        try:
            # Ensure the model is pulled
            ollama.pull(model_name)
            return model_name
        except Exception as e:
            logger.error(f"Failed to initialize Phi model: {str(e)}")
            return None

    def initialize_gemma_model(self, model_name="gemma2:2b"):
        try:
            # Ensure the model is pulled
            ollama.pull(model_name)
            return model_name
        except Exception as e:
            logger.error(f"Failed to initialize Gemma model: {str(e)}")
            return None
        
    def initialize_gemma3_model(self, model_name="gemma3:1b"):
        try:
        # Ensure the model is pulled
            ollama.pull(model_name)
            return model_name
        except Exception as e:
            logger.error(f"Failed to initialize Gemma model: {str(e)}")
            return None

    def init_models(self):
        self.models['qwen'] = self.initialize_qwen_model()
        self.models['llama'] = self.initialize_llama_model()
        self.models['phi'] = self.initialize_phi_model()
        self.models['gemma'] = self.initialize_gemma_model()
        self.models['gemma3'] = self.initialize_gemma3_model()
        self.models['deepseekcoder'] = self.initialize_deepseek_coder_model()

        # Set the first available model as active
        for model_name, model in self.models.items():
            if model is not None:
                self.set_active_model(model_name)
                break

        if self.active_model is None:
            raise ValueError("No models were successfully initialized")

    def set_active_model(self, model_name):
        if model_name in self.models and self.models[model_name] is not None:
            self.active_model = model_name
        else:
            raise ValueError(f"Model {model_name} not found or not initialized")

    def get_active_model(self):
        if self.active_model is None:
            raise ValueError("No active model set")
        return self.models[self.active_model]

def get_model_manager():
    return ModelManager()

def llm_generate_response(model_name, query_plus_context, max_length=200, temperature=float) -> str:
    response = ollama.generate(model=model_name, 
                               prompt=query_plus_context,
                               options={
                                   'temperature': temperature,
                                   'num_predict':500,
                                }
                               )
    # Extract only the generated answer
    # answer = f"Answer from model {model_name}\r\n"
    # answer = answer + response['response'].strip()
    answer =response['response'].strip()
    logger.info(f"model temperature: {temperature}")
    logger.info(f"Raw Ollama response using model {model_name}: {response}")
    return answer



