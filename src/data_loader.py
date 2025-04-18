import os
import json
import random
import glob

class MTDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_combined_dataset(self, num_samples=150, domain_filter=None):
        """
        Load and combine data from all available sources, removing duplicates.
        """
        combined_data = {}
        
        content_hash = set()
        
        domains = ["medical", "technical", "legal", "general"]
        
        if domain_filter:
            if isinstance(domain_filter, str):
                domains = [domain_filter]
            else:
                domains = domain_filter
        
        for domain in domains:
            domain_dir = os.path.join(self.data_dir, domain)
            if not os.path.exists(domain_dir):
                print(f"Warning: Domain directory {domain_dir} not found")
                continue
                
            json_files = glob.glob(os.path.join(domain_dir, "*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        domain_data = json.load(f)
                        
                        for sample_id, sample in domain_data.items():
                            if isinstance(sample, dict) and "source" in sample:
                                content = sample["source"]
                                content_key = content[:100].lower()
                                
                                if content_key not in content_hash:
                                    content_hash.add(content_key)
                                    combined_data[sample_id] = sample
                except Exception as e:
                    print(f"Error loading {json_file}: {str(e)}")
        
        youtube_file = os.path.join(self.data_dir, "youtube_videos.json")
        if os.path.exists(youtube_file):
            try:
                with open(youtube_file, 'r', encoding='utf-8') as f:
                    videos = json.load(f)
                    for i, video in enumerate(videos):
                        video_id = video.get("video_id", f"youtube_{i}")
                        content = video.get("captions", "")
                        if isinstance(content, list):
                            content = " ".join(content)
                            
                        content_key = content[:100].lower()
                        
                        if content_key not in content_hash:
                            content_hash.add(content_key)
                            combined_data[video_id] = {
                                "source": content,
                                "domain": "general",
                                "reference": video.get("gold_transcript", "")
                            }
            except Exception as e:
                print(f"Error loading YouTube videos: {str(e)}")
        
        processed_file = os.path.join(self.data_dir, "processed_videos.json")
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    videos = json.load(f)
                    for i, video in enumerate(videos):
                        video_id = video.get("video_id", f"processed_{i}")
                        content = video.get("captions", "")
                        if isinstance(content, list):
                            content = " ".join(content)
                            
                        content_key = content[:100].lower()
                        
                        if content_key not in content_hash:
                            content_hash.add(content_key)
                            combined_data[video_id] = {
                                "source": content,
                                "domain": "general",
                                "reference": video.get("gold_transcript", "")
                            }
            except Exception as e:
                print(f"Error loading processed videos: {str(e)}")
        
        print(f"Total unique samples loaded: {len(combined_data)}")
        
        if num_samples and len(combined_data) > num_samples:
            sample_ids = list(combined_data.keys())
            selected_ids = random.sample(sample_ids, num_samples)
            combined_data = {k: combined_data[k] for k in selected_ids}
            print(f"Limited to {num_samples} random samples")
            
        return combined_data
