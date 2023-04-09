
import gdown
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_weights():
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)

    f_checkpoint = Path("model/scripted_model1.pt")

    # https://drive.google.com/file/d/1j-ECpr0PIVbJGeRKFeYozDq7M4urk7sP/view?usp=share_link
    if not f_checkpoint.exists():
        # from GD_download import download_file_from_google_drive
        # download_file_from_google_drive(cloud_model_location, f_checkpoint)
        url = 'https://drive.google.com/uc?id=1j-ECpr0PIVbJGeRKFeYozDq7M4urk7sP'
        out = "model/scripted_model1.pt"
        gdown.download(url, out, quiet=False)

    model = torch.jit.load(f_checkpoint)

    return f_checkpoint

