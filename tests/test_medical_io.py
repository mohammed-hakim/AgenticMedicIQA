"""Tests for medical image ingestion and privacy helpers."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.agentic.medical_io import (
    DICOM_EXTENSIONS,
    NATURAL_IMAGE_EXTENSIONS,
    NIFTI_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    get_display_image,
    get_display_path,
    get_medical_metadata,
    infer_modality_hint,
    is_dicom,
    is_medical_image,
    is_nifti,
)
from src.agentic.privacy import redact_metadata, redact_path, redact_text


class TestFormatDetection:
    def test_extension_sets(self):
        assert ".dcm" in DICOM_EXTENSIONS
        assert ".nii.gz" in NIFTI_EXTENSIONS
        assert ".jpg" in NATURAL_IMAGE_EXTENSIONS
        assert DICOM_EXTENSIONS <= SUPPORTED_IMAGE_EXTENSIONS
        assert NIFTI_EXTENSIONS <= SUPPORTED_IMAGE_EXTENSIONS

    def test_detection_is_case_insensitive(self):
        assert is_dicom("STUDY.DCM")
        assert is_dicom("study.dicom")
        assert is_nifti("brain.NII.GZ")
        assert is_nifti("brain.nii")
        assert is_medical_image("brain.nii.gz")
        assert not is_medical_image("photo.jpg")


class TestRasterMetadata:
    def test_raster_metadata(self, tmp_path):
        path = tmp_path / "image.png"
        Image.new("RGB", (64, 48), color=(10, 20, 30)).save(path)

        metadata = get_medical_metadata(str(path))
        assert metadata["source_format"] == "png"
        assert metadata["rows"] == 48
        assert metadata["columns"] == 64
        assert metadata["is_medical"] is False

    def test_display_path_is_unchanged_for_raster(self, tmp_path):
        path = tmp_path / "image.png"
        Image.new("RGB", (32, 32)).save(path)
        assert get_display_path(str(path)) == str(path)

    def test_display_image_resizes_and_warns(self, tmp_path, capsys):
        path = tmp_path / "large.png"
        Image.new("RGB", (800, 400)).save(path)
        image = get_display_image(str(path), max_size=200)
        assert max(image.size) <= 200
        assert "Warning: Resized image" in capsys.readouterr().out

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported image format"):
            get_display_image(str(path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Image file not found"):
            get_display_image(str(tmp_path / "missing.jpg"))


class TestModalityHint:
    def test_dicom_codes(self):
        assert infer_modality_hint("CT") == "ct"
        assert infer_modality_hint("MR") == "mri"
        assert infer_modality_hint("US") == "ultrasound"
        assert infer_modality_hint("MG") == "mammography"
        assert infer_modality_hint("CR") == "radiography"
        assert infer_modality_hint("DX") == "radiography"

    def test_body_part_fallback(self):
        assert infer_modality_hint(None, "BREAST") == "mammography"
        assert infer_modality_hint("OT", "SKIN") == "dermatology"

    def test_unknown_returns_none(self):
        assert infer_modality_hint("ZZZ", "UNKNOWN") is None


class TestPrivacy:
    def test_redact_path_hides_directory_and_name(self):
        redacted = redact_path(r"C:\hospital\patient_123\scan.dcm")
        assert "hospital" not in redacted
        assert "patient_123" not in redacted
        assert redacted.endswith(".dcm")

    def test_redact_nifti_suffix(self):
        assert redact_path("/data/brain.nii.gz").endswith(".nii.gz")

    def test_redact_text_replaces_absolute_paths(self):
        text = r"Failed for C:\data\patient.dcm and /data/patient2.png"
        redacted = redact_text(text)
        assert "patient" not in redacted
        assert redacted.count("<redacted-path>") == 2

    def test_redact_metadata_drops_identifiers(self):
        clean = redact_metadata(
            {
                "Modality": "CT",
                "PatientName": "Doe^Jane",
                "PatientID": "12345",
                "InstitutionName": "Example Hospital",
                "Rows": 512,
            }
        )
        assert clean == {"Modality": "CT", "Rows": 512}


class TestDicomRendering:
    def test_dicom_round_trip(self, tmp_path, monkeypatch):
        pydicom = pytest.importorskip("pydicom")
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

        monkeypatch.setenv("AGENTIC_RENDER_CACHE", str(tmp_path / "cache"))

        arr = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        dataset = FileDataset(
            str(tmp_path / "test.dcm"), {}, file_meta=meta, preamble=b"\0" * 128
        )
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.Modality = "CT"
        dataset.BodyPartExamined = "ABDOMEN"
        dataset.PatientName = "Test^Patient"
        dataset.Rows = 32
        dataset.Columns = 32
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleSlope = 1
        dataset.RescaleIntercept = -1024
        dataset.WindowCenter = 40
        dataset.WindowWidth = 400
        dataset.PixelData = arr.tobytes()

        path = tmp_path / "test.dcm"
        dataset.save_as(path)

        metadata = get_medical_metadata(str(path))
        assert metadata["source_format"] == "dicom"
        assert metadata["modality_code"] == "CT"
        assert metadata["modality_hint"] == "ct"
        assert metadata["rows"] == 32
        assert "PatientName" not in metadata

        image = get_display_image(str(path))
        assert image.mode == "RGB"
        assert image.size == (32, 32)

        rendered = Path(get_display_path(str(path)))
        assert rendered.suffix == ".png"
        assert rendered.exists()

    def test_monochrome1_is_inverted(self, tmp_path, monkeypatch):
        pydicom = pytest.importorskip("pydicom")
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

        monkeypatch.setenv("AGENTIC_RENDER_CACHE", str(tmp_path / "cache"))

        arr = np.zeros((8, 8), dtype=np.uint8)
        arr[0, 0] = 255
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        dataset = FileDataset(
            str(tmp_path / "mono1.dcm"), {}, file_meta=meta, preamble=b"\0" * 128
        )
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.Modality = "DX"
        dataset.Rows = 8
        dataset.Columns = 8
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME1"
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0
        dataset.PixelData = arr.tobytes()

        path = tmp_path / "mono1.dcm"
        dataset.save_as(path)

        image = get_display_image(str(path))
        pixels = np.asarray(image)
        assert pixels[0, 0, 0] == 0

    def test_dicom_without_pydicom_raises_helpful_error(self, tmp_path, monkeypatch):
        import builtins

        path = tmp_path / "fake.dcm"
        path.write_bytes(b"not a real dicom")

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pydicom" or name.startswith("pydicom."):
                raise ImportError("pydicom is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ValueError, match="pydicom"):
            get_display_image(str(path))


class TestNiftiRendering:
    def test_nifti_round_trip(self, tmp_path, monkeypatch):
        nib = pytest.importorskip("nibabel")
        monkeypatch.setenv("AGENTIC_RENDER_CACHE", str(tmp_path / "cache"))

        data = np.arange(16 * 16 * 8, dtype=np.float32).reshape(16, 16, 8)
        volume = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "brain.nii.gz"
        nib.save(volume, str(path))

        metadata = get_medical_metadata(str(path))
        assert metadata["source_format"] == "nifti"
        assert metadata["shape"][:3] == [16, 16, 8]
        assert metadata["is_medical"] is True

        image = get_display_image(str(path))
        assert image.mode == "RGB"
        assert image.size == (16, 16)

        rendered = Path(get_display_path(str(path)))
        assert rendered.suffix == ".png"
        assert rendered.exists()
