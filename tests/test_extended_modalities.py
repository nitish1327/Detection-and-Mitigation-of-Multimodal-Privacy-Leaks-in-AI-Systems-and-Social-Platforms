import csv
import io
from pathlib import Path
import docx
import numpy as np
import pytest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from leakwatch.orchestration.pipeline import PipelineManager
from leakwatch.detection.image import ImageDetector
from leakwatch.mitigation.audio import AudioMitigator
from leakwatch.utils.config import LeakWatchConfig, AppConfig, AudioConfig, ImageConfig, TextConfig
from leakwatch.utils.types import DetectedEntity, Modality, Span, BoundingBox


@pytest.fixture
def test_pipeline(tmp_path: Path) -> PipelineManager:
    from leakwatch.utils.config import get_config
    base_cfg = get_config()
    cfg = base_cfg.model_copy(deep=True)
    cfg.app.output_dir = tmp_path / "artifacts"
    cfg.image.enable_ocr = False
    return PipelineManager(cfg)


def test_process_csv(test_pipeline: PipelineManager, tmp_path: Path):
    csv_file = tmp_path / "contacts.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Email", "Phone"])
        writer.writerow(["John Doe", "john.doe@example.com", "+1-415-555-0199"])
        writer.writerow(["Aarav Sharma", "aarav@example.in", "+91 9876543210"])

    res = test_pipeline.process_text(csv_file)
    assert res.mitigated_output is not None
    assert res.mitigated_output.exists()
    assert len(res.entities) > 0

    content = res.mitigated_output.read_text(encoding="utf-8")
    assert "john.doe@example.com" not in content
    assert "+91 9876543210" not in content


def test_process_docx(test_pipeline: PipelineManager, tmp_path: Path):
    docx_file = tmp_path / "confidential.docx"
    doc = docx.Document()
    doc.add_paragraph("Customer Name: Jane Smith")
    doc.add_paragraph("Contact email: jane.smith@secretcorp.com and phone +1 555-0144")
    doc.save(str(docx_file))

    res = test_pipeline.process_text(docx_file)
    assert res.mitigated_output is not None
    assert res.mitigated_output.exists()
    assert len(res.entities) > 0

    sanitized_doc = docx.Document(str(res.mitigated_output))
    full_text = " ".join(p.text for p in sanitized_doc.paragraphs)
    assert "jane.smith@secretcorp.com" not in full_text


def test_process_pdf(test_pipeline: PipelineManager, tmp_path: Path):
    pdf_file = tmp_path / "statement.pdf"
    c = canvas.Canvas(str(pdf_file), pagesize=letter)
    c.drawString(100, 700, "Account Holder: Robert Johnson")
    c.drawString(100, 680, "Email address: robert.j@bankexample.com")
    c.drawString(100, 660, "Emergency number: +1 415-555-0122")
    c.save()

    res = test_pipeline.process_text(pdf_file)
    assert res.mitigated_output is not None
    assert res.mitigated_output.exists()
    assert len(res.entities) > 0


def test_process_log(test_pipeline: PipelineManager, tmp_path: Path):
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "2026-10-22 10:00:00 INFO User login from IP 192.168.1.50 by user alice@example.com\n"
        "2026-10-22 10:01:00 WARN Failed auth for bob@example.com from 10.0.0.12\n",
        encoding="utf-8"
    )

    res = test_pipeline.process_text(log_file)
    assert res.mitigated_output is not None
    assert res.mitigated_output.exists()
    content = res.mitigated_output.read_text(encoding="utf-8")
    assert "alice@example.com" not in content
    assert "bob@example.com" not in content


def test_process_rtf(test_pipeline: PipelineManager, tmp_path: Path):
    rtf_file = tmp_path / "doc.rtf"
    rtf_file.write_text(
        r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Courier;}}\f0\fs20 Employee: David Miller\par Contact: david.miller@corp.com\par}",
        encoding="utf-8"
    )

    res = test_pipeline.process_text(rtf_file)
    assert res.mitigated_output is not None
    assert res.mitigated_output.exists()


def test_audio_masking_styles(tmp_path: Path):
    import wave
    # Create simple 1-sec mono 16-bit 16kHz wav
    source_wav = tmp_path / "test_input.wav"
    with wave.open(str(source_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        # 16000 samples of 440 Hz tone
        samples = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 10000).astype(np.int16)
        w.writeframes(samples.tobytes())

    entities = [
        DetectedEntity(
            modality=Modality.AUDIO,
            label="EMAIL",
            text="secret@example.com",
            confidence=1.0,
            span=Span(start=10, end=28)
        )
    ]
    transcript = "Contact at secret@example.com for info"

    for style in ["silence", "white_noise", "pink_noise", "garble"]:
        cfg = AudioConfig(mask_style=style)
        mitigator = AudioMitigator(cfg)
        out_wav = tmp_path / f"out_{style}.wav"
        res_path, ranges = mitigator.mitigate(source_wav, transcript, entities, out_wav)
        assert res_path.exists()
        assert res_path.stat().st_size > 0


def test_image_detectors(tmp_path: Path):
    cfg = ImageConfig(enable_ocr=False)
    detector = ImageDetector(cfg)

    # 1. Barcode detector test
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8) + 255
    # draw vertical bars
    for x in range(50, 150, 6):
        blank_img[50:120, x:x+3] = 0
    barcodes = detector._detect_barcodes(blank_img)
    assert isinstance(barcodes, list)

    # 2. License plate visual detector test
    plate_img = np.zeros((400, 600, 3), dtype=np.uint8) + 200
    # draw a rectangular plate with border
    plate_img[150:210, 200:400] = 255
    plate_img[150:153, 200:400] = 0
    plate_img[207:210, 200:400] = 0
    plate_img[150:210, 200:203] = 0
    plate_img[150:210, 397:400] = 0
    # simulate text characters inside
    for x in range(220, 380, 18):
        plate_img[165:195, x:x+8] = 0
    plates = detector._detect_license_plates(plate_img)
    assert isinstance(plates, list)

    # 3. Signature / stamp detector test with blue ink
    doc_img = np.zeros((500, 500, 3), dtype=np.uint8) + 255
    # draw blue oval stamp: in BGR, blue is (200, 50, 20)
    import cv2
    cv2.circle(doc_img, (250, 250), 40, (180, 40, 20), -1)
    stamps = detector._detect_signatures_and_stamps(doc_img)
    assert isinstance(stamps, list)
    assert len(stamps) >= 1
    assert stamps[0].label in {"STAMP", "SIGNATURE"}
