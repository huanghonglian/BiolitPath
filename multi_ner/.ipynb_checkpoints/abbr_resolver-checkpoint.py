import subprocess
import tempfile
import os

class Abbr_resolver():
    def __init__(self, ab3p_path):
        self.ab3p_path = ab3p_path
        
    def resolve(self, text_content):
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(text_content)
            temp_file_path = temp_file.name
        
        try:
            # Run Ab3P to handle temporary files
            result = subprocess.run([self.ab3p_path, temp_file_path], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            line = result.stdout.decode('utf-8')
            if "Path file for type cshset does not exist!" in line:
                raise Exception(line)
            elif "Cannot open" in line:
                raise Exception(line)
            elif "failed to open" in line:
                raise Exception(line)
            
            lines = line.split("\n")
            result_dict = {}
            for line in lines:
                if len(line.split("|")) == 3:
                    sf, lf, score = line.split("|")
                    sf = sf.strip()
                    lf = lf.strip()
                    score=score.strip()
                    if score[0:2]!='0.' or len(score)>9:
                        continue
                    result_dict[sf] = lf
            return result_dict
            
        finally:
            # Delete temporary files
            try:
                os.unlink(temp_file_path)
            except:
                pass
                