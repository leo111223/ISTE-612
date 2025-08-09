import pandas as pd
import os
from llama_client import summarize_definition, client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re

# Try to import transformers for BERT, but make it optional
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. BERT features will be disabled.")

class MLEnhancedSlangDictionary:
    def __init__(self, data_dir='urbandict'):
        
        # Load all datasets
        try:
            self.slang_to_standard_df = pd.read_csv(os.path.join(data_dir, 'slangbridge_complete_dataset.csv'))
            
        except FileNotFoundError:
            print("Main dataset not found, continuing with translation files only")
            self.slang_to_standard_df = pd.DataFrame()
        
        try:
            self.translation_df1 = pd.read_csv(os.path.join(data_dir, 'slang_translation2.csv'))
           
        except FileNotFoundError:
            print("slang_translation2.csv not found")
            self.translation_df1 = pd.DataFrame()
        
        try:
            self.translation_df2 = pd.read_csv(os.path.join(data_dir, 'slang_corpus_chunk_1_to_4.csv'))

        except FileNotFoundError:
            print("slang_corpus_chunk_1_to_4.csv not found")
            self.translation_df2 = pd.DataFrame()
        
        # Prepare data for bidirectional lookup
        self._prepare_datasets()
        
        # Initialize ML models
       
        self._initialize_ml_models()
        
        # Initialize BERT for contextual detection
 
        self._initialize_bert()
        
        # Initialize TF-IDF vectorizer for similarity matching
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
        self._build_similarity_index()
    
    def _prepare_datasets(self):
        #Prepare and combine datasets for bidirectional translation
       
        
        # Combine translation datasets
        translation_dfs = []
        
        if not self.translation_df1.empty:
            df1 = self.translation_df1.copy()
            if 'Standard' in df1.columns and 'Slang' in df1.columns:
                df1.columns = ['standard', 'slang']
            elif len(df1.columns) >= 2:
                df1.columns = ['standard', 'slang'] + list(df1.columns[2:])
            translation_dfs.append(df1[['standard', 'slang']])
        
        if not self.translation_df2.empty:
            df2 = self.translation_df2.copy()
            if 'Standard' in df2.columns and 'Slang' in df2.columns:
                df2.columns = ['standard', 'slang']
            elif len(df2.columns) >= 2:
                df2.columns = ['standard', 'slang'] + list(df2.columns[2:])
            translation_dfs.append(df2[['standard', 'slang']])
        
        if translation_dfs:
            combined_translations = pd.concat(translation_dfs, ignore_index=True)
            
            # Clean and normalize data
            combined_translations['standard'] = combined_translations['standard'].astype(str).str.lower().str.strip()
            combined_translations['slang'] = combined_translations['slang'].astype(str).str.lower().str.strip()
            
            # Remove rows with nan, empty strings, or 'nan' strings
            combined_translations = combined_translations[
                (combined_translations['standard'] != 'nan') & 
                (combined_translations['slang'] != 'nan') &
                (combined_translations['standard'].str.len() > 0) &
                (combined_translations['slang'].str.len() > 0)
            ]
            
            # Remove duplicates
            combined_translations = combined_translations.drop_duplicates()
            
            # Create bidirectional lookup dictionaries
            self.slang_to_standard = dict(zip(combined_translations['slang'], combined_translations['standard']))
            self.standard_to_slang = dict(zip(combined_translations['standard'], combined_translations['slang']))
            
            # Store combined data for ML training
            self.combined_data = combined_translations
        else:
            self.slang_to_standard = {}
            self.standard_to_slang = {}
            self.combined_data = pd.DataFrame()
        
        # Prepare original dataset
        if not self.slang_to_standard_df.empty and 'slang_term' in self.slang_to_standard_df.columns:
            self.slang_to_standard_df['slang_term'] = self.slang_to_standard_df['slang_term'].astype(str).str.lower().str.strip()
        
        
    def _initialize_ml_models(self):
        #Initialize text classification models 
        try:
            # Create training data for text classification
            training_texts = []
            training_labels = []
            
            # Add slang terms with label 'slang'
            for slang_term in self.slang_to_standard.keys():
                training_texts.append(slang_term)
                training_labels.append('slang')
            
            # Add standard terms with label 'standard'
            for standard_term in self.standard_to_slang.keys():
                training_texts.append(standard_term)
                training_labels.append('standard')
            
            if len(training_texts) > 10:  # Need enough data to train
                # Vectorize the text data
                self.classification_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
                X = self.classification_vectorizer.fit_transform(training_texts)
                
                # Encode labels
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(training_labels)
                
                # Split data for training
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train Naive Bayes classifier
                self.nb_classifier = MultinomialNB()
                self.nb_classifier.fit(X_train, y_train)
                
                # Train Logistic Regression classifier
                self.lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
                self.lr_classifier.fit(X_train, y_train)
                
                # Calculate accuracy
                nb_accuracy = self.nb_classifier.score(X_test, y_test)
                lr_accuracy = self.lr_classifier.score(X_test, y_test)
                
                print(f" \nNaive Bayes classifier trained (Accuracy: {nb_accuracy:.3f})")
                print(f" \nLogistic Regression classifier trained (Accuracy: {lr_accuracy:.3f})")
                
                self.classifiers_trained = True
            else:
                print(" Not enough data to train classifiers")
                self.classifiers_trained = False
                
        except Exception as e:
            print(f" Error training classifiers: {e}")
            self.classifiers_trained = False
    
    def _initialize_bert(self):
        #Initialize BERT model for contextual detection
        if not TRANSFORMERS_AVAILABLE:
            print(" Transformers not available, skipping BERT initialization")
            self.bert_available = False
            return
            
        try:
            # Use a lightweight BERT model for sentiment/context analysis
            self.bert_sentiment = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                         return_all_scores=True)
            
            self.bert_available = True
        except Exception as e:
            print(f" Could not load BERT model: {e}")
            
            self.bert_available = False
    
    def _build_similarity_index(self):
        """Build TF-IDF index for cosine similarity matching"""
       
        try:
            # Combine all terms and definitions for similarity matching
            all_terms = []
            all_definitions = []
            
            # From original dataset
            if not self.slang_to_standard_df.empty and 'slang_term' in self.slang_to_standard_df.columns and 'standard_translation' in self.slang_to_standard_df.columns:
                valid_rows = self.slang_to_standard_df.dropna(subset=['slang_term', 'standard_translation'])
                all_terms.extend(valid_rows['slang_term'].tolist())
                all_definitions.extend(valid_rows['standard_translation'].astype(str).tolist())
            
            # From translation datasets
            all_terms.extend(list(self.slang_to_standard.keys()))
            all_definitions.extend(list(self.slang_to_standard.values()))
            
            # Create corpus for TF-IDF
            if all_terms and all_definitions:
                self.corpus = [f"{term} {definition}" for term, definition in zip(all_terms, all_definitions)]
                self.term_list = all_terms
                
                if len(self.corpus) > 0:
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
                    
                else:
                    self.tfidf_matrix = None
                    print(" No valid data for similarity index")
            else:
                self.tfidf_matrix = None
                self.corpus = []
                self.term_list = []
                print(" No data available for similarity index")
                
        except Exception as e:
            print(f" Error building TF-IDF index: {e}")
            self.tfidf_matrix = None
            self.corpus = []
            self.term_list = []
    
    def _find_similar_terms_tfidf(self, query, max_results=3, threshold=0.1):
        """Find similar terms using TF-IDF and cosine similarity"""
        if self.tfidf_matrix is None or len(self.corpus) == 0:
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query.lower()])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top similar terms above threshold
            similar_indices = np.argsort(similarities)[::-1]
            results = []
            
            for idx in similar_indices[:max_results]:
                if idx < len(self.term_list) and similarities[idx] > threshold:
                    results.append({
                        'term': self.term_list[idx],
                        'similarity': similarities[idx],
                        'method': 'tfidf_cosine'
                    })
            
            return results
        except Exception as e:
            print(f" Error in TF-IDF similarity search: {e}")
            return []
    
    def _classify_text_type(self, text):
        """Use ML classifiers to determine if text is slang or standard"""
        if not self.classifiers_trained:
            return self._detect_input_type_heuristic(text)
        
        try:
            # Vectorize the input text
            text_vector = self.classification_vectorizer.transform([text.lower()])
            
            # Get predictions from both classifiers
            nb_pred = self.nb_classifier.predict(text_vector)[0]
            lr_pred = self.lr_classifier.predict(text_vector)[0]
            
            # Get prediction probabilities
            nb_proba = self.nb_classifier.predict_proba(text_vector)[0]
            lr_proba = self.lr_classifier.predict_proba(text_vector)[0]
            
            # Decode predictions
            nb_label = self.label_encoder.inverse_transform([nb_pred])[0]
            lr_label = self.label_encoder.inverse_transform([lr_pred])[0]
            
            # Use ensemble approach - if both agree, high confidence
            if nb_label == lr_label:
                confidence = 'high'
                result = nb_label
            else:
                confidence = 'medium'
                # Use the classifier with higher probability
                nb_max_proba = np.max(nb_proba)
                lr_max_proba = np.max(lr_proba)
                result = nb_label if nb_max_proba > lr_max_proba else lr_label
            
            return {
                'type': result,
                'confidence': confidence,
                'nb_prediction': nb_label,
                'lr_prediction': lr_label,
                'method': 'ml_classification'
            }
            
        except Exception as e:
            print(f" Error in ML classification: {e}")
            return self._detect_input_type_heuristic(text)
    
    def _detect_input_type_heuristic(self, text):
        """Fallback heuristic method for input type detection"""
        slang_indicators = ['ur', 'u', 'wat', 'da', 'dis', 'dat', 'tha', 'n', 'w/', 'b4', 'gr8', '2', '4u']
        standard_indicators = ['the', 'and', 'that', 'with', 'have', 'this', 'will', 'you', 'are', 'for']
        
        words = text.lower().split()
        slang_count = sum(1 for word in words if any(indicator in word for indicator in slang_indicators))
        standard_count = sum(1 for word in words if word in standard_indicators)
        
        if slang_count > standard_count:
            return {'type': 'slang', 'confidence': 'medium', 'method': 'heuristic'}
        elif standard_count > slang_count:
            return {'type': 'standard', 'confidence': 'medium', 'method': 'heuristic'}
        else:
            return {'type': 'mixed', 'confidence': 'low', 'method': 'heuristic'}
    
    def _detect_context_bert(self, text):
        """Use BERT for advanced contextual detection"""
        if not self.bert_available or not text:
            return self._detect_context_heuristic(text)
        
        try:
            # Get sentiment analysis from BERT
            results = self.bert_sentiment(text)
            
            contexts = []
            for result in results[0]:  # results is a list with one element containing all scores
                label = result['label'].lower()
                score = result['score']
                
                if score > 0.6:  # High confidence threshold
                    if 'positive' in label:
                        contexts.append('positive')
                    elif 'negative' in label:
                        contexts.append('negative')
                    elif 'neutral' in label:
                        contexts.append('neutral')
            
            # Add heuristic context detection
            heuristic_contexts = self._detect_context_heuristic(text)
            contexts.extend([ctx for ctx in heuristic_contexts if ctx not in contexts])
            
            return contexts if contexts else ['neutral']
            
        except Exception as e:
            print(f"Error in BERT context detection: {e}")
            return self._detect_context_heuristic(text)
    
    def _detect_context_heuristic(self, text):
        """Heuristic context detection based on keywords"""
        if not text:
            return ['neutral']
            
        text_lower = text.lower()
        
        contexts = {
            'positive': ['good', 'great', 'awesome', 'love', 'like', 'happy', 'excited', 'amazing', 'cool', 'nice'],
            'negative': ['bad', 'hate', 'angry', 'sad', 'terrible', 'awful', 'mad', 'sucks', 'horrible'],
            'casual': ['lol', 'omg', 'tbh', 'ngl', 'fr', 'rn', 'wyd', 'bruh', 'yo', 'hey'],
            'formal': ['however', 'therefore', 'furthermore', 'nevertheless', 'moreover', 'consequently']
        }
        
        detected_contexts = []
        for context, keywords in contexts.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_contexts.append(context)
        
        return detected_contexts if detected_contexts else ['neutral']
    
    def translate_slang_to_standard(self, term, context=""):
        """Enhanced translation with ML techniques"""
        if not term:
            return None
            
        term_clean = term.lower().strip()
        
        # 1. Direct lookup in combined translation data
        if term_clean in self.slang_to_standard:
            translation = self.slang_to_standard[term_clean]
            contexts = self._detect_context_bert(context) if context else ['neutral']
            return {
                'translation': translation,
                'method': 'direct_lookup',
                'context': contexts,
                'confidence': 'high'
            }
        
        # 2. Try original dataset
        if not self.slang_to_standard_df.empty and 'slang_term' in self.slang_to_standard_df.columns:
            row = self.slang_to_standard_df[self.slang_to_standard_df['slang_term'] == term_clean]
            if not row.empty:
                definition = row.iloc[0]['standard_translation']
                contexts = self._detect_context_bert(context) if context else ['neutral']
                return {
                    'translation': str(definition),
                    'method': 'dataset_lookup',
                    'context': contexts,
                    'confidence': 'high'
                }
        
        # 3. Try TF-IDF cosine similarity matching
        similar_terms = self._find_similar_terms_tfidf(term_clean)
        if similar_terms:
            best_match = similar_terms[0]
            if best_match['term'] in self.slang_to_standard:
                translation = self.slang_to_standard[best_match['term']]
                contexts = self._detect_context_bert(context) if context else ['neutral']
                return {
                    'translation': translation,
                    'method': 'tfidf_cosine_similarity',
                    'context': contexts,
                    'confidence': 'medium',
                    'similar_term': best_match['term'],
                    'similarity_score': best_match['similarity']
                }
        
        return None
    
    def translate_standard_to_slang(self, phrase, context=""):
        """Enhanced standard to slang translation with ML techniques"""
        if not phrase:
            return None
            
        phrase_clean = phrase.lower().strip()
        
        # 1. Direct lookup
        if phrase_clean in self.standard_to_slang:
            translation = self.standard_to_slang[phrase_clean]
            contexts = self._detect_context_bert(context) if context else ['neutral']
            return {
                'translation': translation,
                'method': 'direct_lookup',
                'context': contexts,
                'confidence': 'high'
            }
        
        # 2. Try TF-IDF similarity matching
        similar_terms = self._find_similar_terms_tfidf(phrase_clean)
        if similar_terms:
            for match in similar_terms:
                if match['term'] in self.standard_to_slang:
                    translation = self.standard_to_slang[match['term']]
                    contexts = self._detect_context_bert(context) if context else ['neutral']
                    return {
                        'translation': translation,
                        'method': 'tfidf_cosine_similarity',
                        'context': contexts,
                        'confidence': 'medium',
                        'similar_term': match['term'],
                        'similarity_score': match['similarity']
                    }
        
        return None
    
    def translate_sentence(self, sentence, direction="auto", context=""):
        """Translate entire sentences with multiple slang terms"""
        if not sentence:
            return {'original': '', 'translated': '', 'changes': [], 'context': ['neutral']}
            
        words = sentence.split()
        translated_words = []
        changes_made = []
        
        for word in words:
            # Clean word (remove punctuation for lookup, but keep for output)
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            
            if direction in ["auto", "slang_to_standard"]:
                result = self.translate_slang_to_standard(clean_word, context)
                if result:
                    # Preserve original capitalization and punctuation
                    original_punct = re.findall(r'[^\w\s]', word)
                    translated_word = result['translation']
                    if word and word[0].isupper():
                        translated_word = translated_word.capitalize()
                    translated_word += ''.join(original_punct)
                    translated_words.append(translated_word)
                    changes_made.append({
                        'original': word,
                        'translated': translated_word,
                        'method': result['method'],
                        'confidence': result['confidence']
                    })
                else:
                    translated_words.append(word)
            
            elif direction == "standard_to_slang":
                result = self.translate_standard_to_slang(clean_word, context)
                if result:
                    # Preserve original capitalization and punctuation
                    original_punct = re.findall(r'[^\w\s]', word)
                    translated_word = result['translation']
                    if word and word[0].isupper():
                        translated_word = translated_word.capitalize()
                    translated_word += ''.join(original_punct)
                    translated_words.append(translated_word)
                    changes_made.append({
                        'original': word,
                        'translated': translated_word,
                        'method': result['method'],
                        'confidence': result['confidence']
                    })
                else:
                    translated_words.append(word)
        
        return {
            'original': sentence,
            'translated': ' '.join(translated_words),
            'changes': changes_made,
            'context': self._detect_context_bert(context) if context else ['neutral']
        }
    
    def analyze_text_with_ml(self, text):
        """Comprehensive ML analysis of input text"""
        # Text type classification
        classification = self._classify_text_type(text)
        
        # Context detection with BERT
        contexts = self._detect_context_bert(text)
        
        # TF-IDF similarity analysis
        similar_terms = self._find_similar_terms_tfidf(text, max_results=5)
        
        return {
            'text': text,
            'classification': classification,
            'contexts': contexts,
            'similar_terms': similar_terms,
            'ml_techniques_used': [
                'TF-IDF + Cosine Similarity',
                'Naive Bayes Classification',
                'Logistic Regression Classification', 
                'BERT Contextual Analysis' if self.bert_available else 'Heuristic Context Analysis'
            ]
        }
    
    def evaluate_translation_accuracy(self, test_samples=50, direction="both"):
        """Test translation accuracy on known pairs"""
        results = {}
        
        # Test slang-to-standard accuracy
        if direction in ["slang_to_standard", "both"]:
            slang_correct = 0
            slang_total = min(test_samples, len(self.slang_to_standard))
            
            if slang_total > 0:
                test_pairs = list(self.slang_to_standard.items())[:slang_total]
                for slang, expected_standard in test_pairs:
                    result = self.translate_slang_to_standard(slang)
                    if result:
                        # Check if translation matches (case-insensitive, flexible matching)
                        translated = result['translation'].lower().strip()
                        expected = expected_standard.lower().strip()
                        if translated == expected or translated in expected or expected in translated:
                            slang_correct += 1
                
                slang_accuracy = slang_correct / slang_total
                results['slang_to_standard'] = {
                    'accuracy': slang_accuracy,
                    'correct': slang_correct,
                    'total': slang_total
                }
        
        # Test standard-to-slang accuracy
        if direction in ["standard_to_slang", "both"]:
            standard_correct = 0
            standard_total = min(test_samples, len(self.standard_to_slang))
            
            if standard_total > 0:
                test_pairs = list(self.standard_to_slang.items())[:standard_total]
                for standard, expected_slang in test_pairs:
                    result = self.translate_standard_to_slang(standard)
                    if result:
                        # Check if translation matches (case-insensitive, flexible matching)
                        translated = result['translation'].lower().strip()
                        expected = expected_slang.lower().strip()
                        if translated == expected or translated in expected or expected in translated:
                            standard_correct += 1
                
                standard_accuracy = standard_correct / standard_total
                results['standard_to_slang'] = {
                    'accuracy': standard_accuracy,
                    'correct': standard_correct,
                    'total': standard_total
                }
        
        return results
    
    def evaluate_classification_performance(self, test_samples=100):
        """Evaluate ML classifier performance"""
        if not self.classifiers_trained:
            return {"error": "Classifiers not trained"}
        
        try:
            # Create test data
            test_texts = []
            true_labels = []
            
            # Add slang samples
            slang_samples = min(test_samples // 2, len(self.slang_to_standard))
            for slang_term in list(self.slang_to_standard.keys())[:slang_samples]:
                test_texts.append(slang_term)
                true_labels.append('slang')
            
            # Add standard samples  
            standard_samples = min(test_samples // 2, len(self.standard_to_slang))
            for standard_term in list(self.standard_to_slang.keys())[:standard_samples]:
                test_texts.append(standard_term)
                true_labels.append('standard')
            
            if len(test_texts) == 0:
                return {"error": "No test data available"}
            
            # Get predictions
            correct_nb = 0
            correct_lr = 0
            
            for text, true_label in zip(test_texts, true_labels):
                # Get ML classification
                classification = self._classify_text_type(text)
                predicted_type = classification['type']
                
                # Check accuracy
                if predicted_type == true_label:
                    if 'nb_prediction' in classification:
                        if classification['nb_prediction'] == true_label:
                            correct_nb += 1
                        if classification['lr_prediction'] == true_label:
                            correct_lr += 1
                    else:
                        correct_nb += 1
                        correct_lr += 1
            
            total = len(test_texts)
            return {
                'naive_bayes_accuracy': correct_nb / total if total > 0 else 0,
                'logistic_regression_accuracy': correct_lr / total if total > 0 else 0,
                'total_samples': total
            }
            
        except Exception as e:
            return {"error": f"Classification evaluation failed: {e}"}
    
    def run_comprehensive_evaluation(self, samples=50):
        """Run complete evaluation suite"""
        print("Running Comprehensive Evaluation...\n")
        
        # Translation accuracy
        translation_results = self.evaluate_translation_accuracy(samples)
        
        print("--TRANSLATION ACCURACY:")
        if 'slang_to_standard' in translation_results:
            s2s = translation_results['slang_to_standard']
            print(f"   Slang → Standard: {s2s['accuracy']:.1%} ({s2s['correct']}/{s2s['total']})")
        
        if 'standard_to_slang' in translation_results:
            std2s = translation_results['standard_to_slang']
            print(f"   Standard → Slang: {std2s['accuracy']:.1%} ({std2s['correct']}/{std2s['total']})")
        
        # Classification performance
        classification_results = self.evaluate_classification_performance(samples * 2)
        
        print("\--*ML CLASSIFICATION ACCURACY:")
        if 'error' not in classification_results:
            print(f"   Naive Bayes: {classification_results['naive_bayes_accuracy']:.1%}")
            print(f"   Logistic Regression: {classification_results['logistic_regression_accuracy']:.1%}")
            print(f"   Test Samples: {classification_results['total_samples']}")
        else:
            print(f"   Error: {classification_results['error']}")
        
        # Overall system stats
        stats = self.get_ml_stats()
        print(f"\n--SYSTEM PERFORMANCE:**")
        print(f"   Dataset Coverage: {stats['slang_to_standard_count'] + stats['standard_to_slang_count']} translations")
        print(f"   TF-IDF Index Size: {stats['tfidf_corpus_size']} entries")
        print(f"   BERT Available: {'' if stats['bert_available'] else ''}")
        print(f"   Classifiers Trained: {'' if stats['classifiers_trained'] else ''}")
        
        return {
            'translation_accuracy': translation_results,
            'classification_accuracy': classification_results,
            'system_stats': stats
        }
    
    def get_ml_stats(self):
        """Get statistics about ML model performance"""
        stats = {
            'slang_to_standard_count': len(self.slang_to_standard),
            'standard_to_slang_count': len(self.standard_to_slang),
            'original_dataset_size': len(self.slang_to_standard_df) if not self.slang_to_standard_df.empty else 0,
            'tfidf_corpus_size': len(self.corpus) if hasattr(self, 'corpus') else 0,
            'classifiers_trained': self.classifiers_trained,
            'bert_available': self.bert_available,
            'ml_techniques': [
                'TF-IDF Vectorization',
                'Cosine Similarity Matching',
                'Naive Bayes Classification',
                'Logistic Regression Classification',
                'BERT Contextual Analysis' if self.bert_available else 'Heuristic Context Analysis'
            ]
        }
        return stats