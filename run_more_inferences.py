import torch
import yaml
import pickle
from transformers import AutoTokenizer
from model import MultiTaskModel
from text_utils import TextCleaner
from example_inference import load_model, predict, phonemize

def run_inferences():
    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT = "model/checkpoint_step_1000000_final.t7"
    CONFIG_PATH = "model/config.yml"
    OUTPUT_FILE = "inference_results.txt"
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    
    TOKEN_MAPS = dataset_params.get("token_maps", "token_maps.pkl")
    TEXT_TOKENIZER = dataset_params.get("tokenizer", "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct")

    print(f"Loading tokenizer: {TEXT_TOKENIZER}")
    llama_tokenizer = AutoTokenizer.from_pretrained(TEXT_TOKENIZER)

    # Load Token Maps
    with open(TOKEN_MAPS, 'rb') as f:
        token_maps = pickle.load(f)
    
    # Reverse map: compact_id -> original_id
    id_to_original = {v['token']: k for k, v in token_maps.items()}
    
    text_cleaner = TextCleaner()
    phoneme_vocab_size = len(text_cleaner.word_index_dictionary)
    bpe_vocab_size = len(token_maps)

    print(f"Phoneme vocab size: {phoneme_vocab_size}")
    print(f"BPE vocab size: {bpe_vocab_size}")

    print(f"Loading model from: {CHECKPOINT}")
    model = load_model(
        CHECKPOINT, 
        phoneme_vocab_size, 
        bpe_vocab_size,
        model_params,
        device=DEVICE
    )

    sentences = [
        "Presiden Joko Widodo meresmikan jembatan baru di Papua kemarin siang.",
        "Batik merupakan warisan budaya Indonesia yang sudah mendunia dan diakui UNESCO.",
        "Nasi goreng adalah makanan favorit banyak orang, baik lokal maupun turis mancanegara.",
        "Gunung Merapi kembali mengeluarkan awan panas pada Selasa pagi, warga diminta waspada.",
        "Teknologi kecerdasan buatan berkembang sangat pesat dalam beberapa tahun terakhir ini.",
        "User interface-nya sangat user-friendly, tapi backend-nya masih perlu banyak optimasi performa.",
        "Bapak SBY (Susilo Bambang Yudhoyono) menjabat sebagai presiden RI ke-6 selama dua periode berturut-turut.",
        "Harga saham gabungan hari ini turun 0,5% menjadi 7.234 poin karena sentimen negatif pasar global.",
        "Ibu membeli 2,5 kg gula, 500 gram kopi, dan 3 liter minyak goreng seharga Rp125.000 di pasar swalayan.",
        "Kakak sedang asyik bermain 'game' di laptop sambil mendengarkan lagu 'heavy metal' menggunakan 'headset'.",
        "Saya suka makan nasi.",
        "Buku itu ada di atas meja.",
        "Kucing saya tidur di kursi.",
        "Adik sedang bermain bola di taman.",
        "Hari ini cuaca sangat cerah."
    ]

    results = []
    
    # Simple output capture for the predict function
    import io
    from contextlib import redirect_stdout

    for i, text in enumerate(sentences):
        print(f"Processing sentence {i+1}/{len(sentences)}...")
        
        # Capture the detailed output of predict
        f_buffer = io.StringIO()
        with redirect_stdout(f_buffer):
            output = predict(model, text, text_cleaner, llama_tokenizer, id_to_original, DEVICE)
        
        predict_log = f_buffer.getvalue()
        results.append({
            "text": text,
            "output": output,
            "log": predict_log
        })

    # Save to txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("PL-BERT-BPE Inference Results\n")
        f.write("="*80 + "\n\n")
        for i, res in enumerate(results):
            f.write(f"Example {i+1}:\n")
            f.write(f"Input: {res['text']}\n")
            f.write(f"Output: {res['output']}\n")
            f.write("-" * 40 + "\n")
            f.write("Detailed Log:\n")
            f.write(res['log'])
            f.write("\n" + "="*80 + "\n\n")

    print(f"✓ All inferences complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inferences()
