from enhanced_slang_lookup import MLEnhancedSlangDictionary
from llama_client import query_llama, query_llama_for_translation

FALLBACK_RESPONSE = "Sorry, I couldn't find a translation for that. "
class SlangBridgeBot:
    def __init__(self):
        #print("Initializing SlangBridge ML Enhanced System...")
        self.slang_dict = MLEnhancedSlangDictionary()
        print("\nSlangBridge is ready!\n")
    
    def detect_input_type(self, text):
        #Detect if input is slang or standard using ML classifiers
        ml_result = self.slang_dict._classify_text_type(text)
        return ml_result['type']
    
    def format_result(self, result, original_text, translation_type):
        #Format translation results for display
        if not result:
            return None
        
        output = f"\n**{translation_type.upper()} TRANSLATION**\n"
        output += f"Original: {original_text}\n"
        output += f"Translation: {result['translation']}\n"
        
        if 'method' in result:
            output += f"Method: {result['method'].replace('_', ' ').title()}\n"
        
        if 'confidence' in result:
            output += f"Confidence: {result['confidence'].title()}\n"
        
        if 'context' in result and result['context'] != ['neutral']:
            output += f"Context: {', '.join(result['context'])}\n"
        
        if 'similarity_score' in result:
            output += f"Similarity Score: {result['similarity_score']:.2f}\n"
        
        return output
    
    def process_input(self, user_input, mode="auto"):
        #Process user input with enhanced bidirectional translation
        user_input = user_input.strip()
        
        if mode == "auto":
            input_type = self.detect_input_type(user_input)
        else:
            input_type = mode
        
        results_found = False
        
        # Try different translation approaches
        if input_type in ["slang", "mixed"] or mode == "slang_to_standard":
            # Slang to Standard Translation
            result = self.slang_dict.translate_slang_to_standard(user_input)
            if result:
                formatted = self.format_result(result, user_input, "Slang to Standard")
                if formatted:
                    print(formatted)
                    results_found = True
        
        if input_type in ["standard", "mixed"] or mode == "standard_to_slang":
            # Standard to Slang Translation
            result = self.slang_dict.translate_standard_to_slang(user_input)
            if result:
                formatted = self.format_result(result, user_input, "Standard to Slang")
                if formatted:
                    print(formatted)
                    results_found = True
        
        # If no local results, try LLAMA API
        if not results_found:
            print("Checking with LLAMA for better translation...")
            try:
                llama_result = query_llama_for_translation(user_input, input_type)
                if llama_result:
                    print(f"**LLAMA TRANSLATION**\n{llama_result}")
                    results_found = True
            except Exception:
                # Fallback to original query_llama
                llama_result = query_llama(user_input)
                if llama_result:
                    print(f"**LLAMA SAYS**\n{llama_result}")
                    results_found = True
        
        if not results_found:
            print(FALLBACK_RESPONSE)
    
    def show_help(self):
        """Display help information"""
        help_text = """
**SlangBridge ML Enhanced - Help**

**Commands:**
- Type any slang term to translate
- 'mode slang' - Switch to slang-to-standard only
- 'mode standard' - Switch to standard-to-slang only  
- 'mode auto' - Auto-detect input type (default)
- 'analyze [text]' - Show detailed ML analysis
- 'stats' - Show ML model statistics
- 'evaluate' - Run comprehensive evaluation
- 'help' - Show this help message
- 'exit' - Quit the program

**Examples:**
- "fire" → translates slang to standard
- "excellent" → translates standard to slang
- "analyze fire" → shows ML analysis

**ML Features:**
TF-IDF + Cosine Similarity matching
Naive Bayes & Logistic Regression classification
BERT contextual analysis
Bidirectional translation (slang ↔ standard)
Context-aware processing
Multiple confidence levels
"""
        print(help_text)
    
    def run(self):
        """Main program loop"""
        print("Welcome to SlangBridge! (Type 'help' for commands, 'exit' to quit)")
        
        
        mode = "auto"  # auto, slang_to_standard, standard_to_slang
        
        while True:
            prompt = f"[{mode}] What can I translate for you?: "
            user_input = input(prompt).strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye! Thanks for using SlangBridge!")
                break
            elif user_input.lower() == 'help':
                self.show_help()
                continue
            elif user_input.lower() == 'stats':
                stats = self.slang_dict.get_ml_stats()
                print("\n **ML Model Statistics:**")
                for key, value in stats.items():
                    if key == 'ml_techniques':
                        print(f"ML Techniques: {', '.join(value)}")
                    else:
                        print(f"   {key.replace('_', ' ').title()}: {value}")
                print()
                continue
            elif user_input.lower() == 'evaluate':
                print("Running evaluation (this may take a moment)...")
                self.slang_dict.run_comprehensive_evaluation()
                print()
                continue
            elif user_input.lower().startswith('analyze '):
                text_to_analyze = user_input[8:].strip()
                if text_to_analyze:
                    analysis = self.slang_dict.analyze_text_with_ml(text_to_analyze)
                    print(f"\n **ML Analysis of: '{text_to_analyze}'**")
                    print(f"Classification: {analysis['classification']['type']} ({analysis['classification']['confidence']} confidence)")
                    print(f"Method: {analysis['classification']['method']}")
                    print(f"Context: {', '.join(analysis['contexts'])}")
                    if analysis['similar_terms']:
                        print("🔗 Similar Terms:")
                        for term in analysis['similar_terms']:
                            print(f"   • {term['term']} (similarity: {term['similarity']:.3f})")
                    print(f"ML Techniques Used: {', '.join(analysis['ml_techniques_used'])}")
                else:
                    print("Please provide text to analyze: analyze [your text]")
                print()
                continue
            elif user_input.lower().startswith('mode '):
                new_mode = user_input.lower().replace('mode ', '').strip()
                if new_mode in ['auto', 'slang', 'standard']:
                    mode = new_mode
                    print(f"Mode switched to: {mode}")
                else:
                    print("Invalid mode. Use: auto, slang, or standard")
                continue
            elif not user_input:
                continue
            
            # Process the translation
            translation_mode = mode if mode != 'auto' else 'auto'
            if mode == 'slang':
                translation_mode = 'slang_to_standard'
            elif mode == 'standard':
                translation_mode = 'standard_to_slang'
            
            self.process_input(user_input, translation_mode)
            print()  # Add spacing between interactions

def main():
    try:
        bot = SlangBridgeBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n Goodbye! Thanks for using SlangBridge!")
    except Exception as e:
        print(f" An error occurred: {e}")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    main()