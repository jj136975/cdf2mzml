from pathlib import Path
from datetime import datetime
import numpy as np
import netCDF4 as nc
import base64
import zlib
import gzip
import logging
from dataclasses import dataclass
from enum import Enum
import os
import traceback

class XMLTag:
    """Helper class for building XML elements"""
    def __init__(self, name: str, buffer: bytearray):
        self.name = name
        self.attributes: dict[str, str] = {}
        self._buffer = buffer
    
    """ Set an attribute for the XML tag """
    def set(self, attr: str, value: str):
        self.attributes[attr] = value
        return self
    
    """ Write the XML tag to the buffer, optionally with text content """
    def write(self, text: str | None = None):
        self._buffer.extend(f"<{self.name}".encode('utf-8'))
        for attr, value in self.attributes.items():
            self._buffer.extend(f' {attr}="{value}"'.encode('utf-8'))
        if text is not None:
            self._buffer.extend(b">")
            self._buffer.extend(text.encode('utf-8'))
            self._buffer.extend(f"</{self.name}>".encode('utf-8'))
        else:
            self._buffer.extend(b"/>")
        return self
        
    def __enter__(self):
        self._buffer.extend(f"<{self.name}".encode('utf-8'))
        for attr, value in self.attributes.items():
            self._buffer.extend(f' {attr}="{value}"'.encode('utf-8'))
        self._buffer.extend(b">")
        return self
        
    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._buffer.extend(f"</{self.name}>".encode('utf-8'))

class XMLBuffer:
    """Helper class for building XML documents"""
    def __init__(self):
        self.buffer = bytearray()
    
    """ Get the current offset in the buffer """
    def get_offset(self) -> int:
        return len(self.buffer)

    """ Create a new XML tag """
    def tag(self, name: str) -> XMLTag:
        return XMLTag(name, self.buffer)


class TaskStatus(Enum):
    """Enumeration of task statuses"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class MassSpectrumType(Enum):
    CENTROIDED = "CENTROIDED"
    PROFILE = "PROFILE"
    THRESHOLDED = "THRESHOLDED"


class PolarityType(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Scan:
    """Represents a single mass spectrum scan"""
    file_name: str
    scan_number: int
    ms_level: int
    retention_time: float
    mz_values: np.ndarray
    intensity_values: np.ndarray
    spectrum_type: MassSpectrumType
    polarity: PolarityType
    scan_definition: str
    tic: float | None = None
    mz_range: tuple[float, float] | None = None

    def __post_init__(self):
        self.mz_values = np.asarray(self.mz_values, dtype=np.float64)
        self.intensity_values = np.asarray(self.intensity_values, dtype=np.float64)
        
        # Calculate TIC if not provided
        if self.tic is None and len(self.intensity_values) > 0:
            self.tic = float(np.sum(self.intensity_values))
        
        # Calculate m/z range if not provided
        if self.mz_range is None and len(self.mz_values) > 0:
            self.mz_range = (float(np.min(self.mz_values)), float(np.max(self.mz_values)))
        
        if not self.scan_definition and self.mz_range:
            self.scan_definition = f"FTMS + p ESI w SIM ms [{self.mz_range[0]:.02f}-{self.mz_range[1]:.02f}]"

class RawDataFile:
    """Container for raw mass spectrometry data"""
    def __init__(self, file_name: str, file_path: Path):
        self.file_name = file_name
        self.file_path = file_path
        self.run_date: datetime | None = None
        self.scans: list[Scan] = []
        self.applied_methods: list[str] = []
    
    def add_scan(self, scan: Scan):
        self.scans.append(scan)
    
    def get_scans(self) -> list[Scan]:
        return self.scans

class CompressionType(Enum):
    NONE = ("MS:1000576", "no compression")
    ZLIB = ("MS:1000574", "zlib compression")
    GZIP = ("MS:1000575", "gzip compression")
    
    def __init__(self, accession, name):
        self.accession = accession
        self._name = name


@dataclass
class CVParam:
    """Controlled vocabulary parameter"""
    accession: str
    name: str = ""
    value: str = ""
    unit_accession: str | None = None


def detect_spectrum_type(mz_values: np.ndarray, intensity_values: np.ndarray) -> MassSpectrumType:
    """
    Detect spectrum type based on analysis of the base peak and surrounding data points.
    Converted from Java ScanUtils.detectSpectrumType method.
    
    Args:
        mz_values: Array of m/z values
        intensity_values: Array of intensity values
        
    Returns:
        MassSpectrumType indicating CENTROIDED, PROFILE, or THRESHOLDED
    """
    # If the spectrum has less than 5 data points, it should be centroided
    if len(mz_values) < 5:
        return MassSpectrumType.CENTROIDED
    
    # Find the base peak index and check for zero data points
    base_peak_index = 0
    has_zero_data_point = False
    
    # Go through the data points and find the highest one
    size = len(mz_values)
    for i in range(size):
        # Update the base_peak_index accordingly
        if intensity_values[i] > intensity_values[base_peak_index]:
            base_peak_index = i
        if intensity_values[i] == 0.0:
            has_zero_data_point = True
    
    # Calculate the total m/z span of the scan
    scan_mz_span = mz_values[size - 1] - mz_values[0]
    
    # Find all data points around the base peak that have intensity above half maximum
    half_intensity = intensity_values[base_peak_index] / 2.0
    
    # Find left boundary
    left_index = base_peak_index
    while left_index > 0 and intensity_values[left_index - 1] > half_intensity:
        left_index -= 1
    
    # Find right boundary
    right_index = base_peak_index
    while right_index < size - 1 and intensity_values[right_index + 1] > half_intensity:
        right_index += 1
    
    # Calculate main feature characteristics
    main_feature_mz_span = mz_values[right_index] - mz_values[left_index]
    main_feature_data_point_count = right_index - left_index + 1
    
    # If the main feature has less than 3 data points above half intensity, it
    # indicates a centroid spectrum. Further, if the m/z span of the main
    # feature is more than 0.1% of the scan m/z range, it also indicates a
    # centroid spectrum. These criteria are empirical and probably not
    # bulletproof. However, it works for all the test cases we have.
    if (main_feature_data_point_count < 3) or (main_feature_mz_span > (scan_mz_span / 1000.0)):
        return MassSpectrumType.CENTROIDED
    else:
        if has_zero_data_point:
            return MassSpectrumType.PROFILE
        else:
            return MassSpectrumType.THRESHOLDED

class MzMLExporter:
    """Exports raw data to mzML format"""
    
    # Constant CV parameters
    CV_PARAMS = {
        'centroided': CVParam("MS:1000127", "centroid mass spectrum"),
        'profile': CVParam("MS:1000128", "profile spectrum"),
        'ms_level': CVParam("MS:1000511", "ms level"),
        'tic': CVParam("MS:1000285", "total ion current"),
        'lowest_mz': CVParam("MS:1000528", "lowest observed m/z", unit_accession="MS:1000040"),
        'highest_mz': CVParam("MS:1000527", "highest observed m/z", unit_accession="MS:1000040"),
        'scan_start_time': CVParam("MS:1000016", "scan time", unit_accession="UO:0000010"),
        'positive_polarity': CVParam("MS:1000130", "positive scan"),
        'negative_polarity': CVParam("MS:1000129", "negative scan"),
        'filter_string': CVParam("MS:1000512", "filter string"),
        'scan_window_lower': CVParam("MS:1000501", "scan window lower limit", unit_accession="MS:1000040"),
        'scan_window_upper': CVParam("MS:1000500", "scan window upper limit", unit_accession="MS:1000040"),
        'mz_array': CVParam("MS:1000514", "m/z array", unit_accession="MS:1000040"),
        'intensity_array': CVParam("MS:1000515", "intensity array", unit_accession="MS:1000131"),
        '64_bit_float': CVParam("MS:1000523", "64-bit float"),
        '32_bit_float': CVParam("MS:1000521", "32-bit float"),
    
        # Additional CV parameters from Java source
        'retention_time': CVParam("MS:1000894", "retention time"),
        'local_retention_time': CVParam("MS:1000895", "local retention time"),
        'normalized_retention_time': CVParam("MS:1000896", "normalized retention time"),
        'ms1_spectrum': CVParam("MS:1000579", "MS1 spectrum"),
        'charge_state': CVParam("MS:1000041", "charge state"),
        'ion_inject_time': CVParam("MS:1000927", "ion injection time"),
        'precursor_mz': CVParam("MS:1000744", "selected ion m/z"),
        'isolation_window_target': CVParam("MS:1000827", "isolation window target m/z"),
        'isolation_window_lower_offset': CVParam("MS:1000828", "isolation window lower offset"),
        'isolation_window_upper_offset': CVParam("MS:1000829", "isolation window upper offset"),
        'activation_energy': CVParam("MS:1000045", "collision energy"),
        'percent_collision_energy': CVParam("MS:1000138", "percent collision energy"),
        'activation_cid': CVParam("MS:1000133", "collision-induced dissociation"),
        'electron_capture_dissociation': CVParam("MS:1000250", "electron capture dissociation"),
        'high_energy_cid': CVParam("MS:1000422", "beam-type collision-induced dissociation"),
        'low_energy_cid': CVParam("MS:1000433", "low-energy collision-induced dissociation"),
        'activation_ead': CVParam("MS:1003294", "electron activated dissociation"),
        'electron_beam_energy_ead': CVParam("MS:1003410", "electron beam energy"),
        'retention_time_array': CVParam("MS:1000595", "time array", unit_accession="UO:0000010"),
        'wavelength_array': CVParam("MS:1000617", "wavelength array", unit_accession="UO:0000018"),
        'uv_spectrum': CVParam("MS:1000804", "electromagnetic radiation spectrum"),
        'fluorescence_detector': CVParam("MS:1002308", "fluorescence detector"),
        'lowest_observed_wavelength': CVParam("MS:1000619", "lowest observed wavelength"),
        'highest_observed_wavelength': CVParam("MS:1000618", "highest observed wavelength"),
        'mobility_drift_time': CVParam("MS:1002476", "ion mobility drift time", unit_accession="UO:0000028"),
        'mobility_inverse_reduced': CVParam("MS:1002815", "inverse reduced ion mobility", unit_accession="MS:1002814"),
    
        # Chromatogram types
        'chromatogram_tic': CVParam("MS:1000235", "total ion current chromatogram"),
        'chromatogram_mrm_srm': CVParam("MS:1001473", "selected reaction monitoring chromatogram"),
        'chromatogram_sim': CVParam("MS:1001472", "selected ion monitoring chromatogram"),
        'chromatogram_sic': CVParam("MS:1000627", "selected ion chromatogram"),
        'chromatogram_bpc': CVParam("MS:1000628", "basepeak chromatogram"),
        'chromatogram_electromagnetic_radiation': CVParam("MS:1000811", "electromagnetic radiation chromatogram"),
        'chromatogram_absorption': CVParam("MS:1000812", "absorption chromatogram"),
        'chromatogram_emission': CVParam("MS:1000813", "emission chromatogram"),
        'chromatogram_ion_current': CVParam("MS:1000810", "ion current chromatogram"),
        'chromatogram_pressure': CVParam("MS:1003019", "pressure chromatogram"),
        'chromatogram_flow_rate': CVParam("MS:1003020", "flow rate chromatogram"),
    }
    
    def __init__(self, raw_data_file: RawDataFile, output_path: Path,
                 compression: CompressionType = CompressionType.ZLIB):
        self.raw_data_file = raw_data_file
        self.output_path = output_path
        self.compression = compression
        self.logger = logging.getLogger(__name__)
        self.cancelled = False
        self.parsed_scans = 0
        self.total_scans = len(raw_data_file.scans)
    
    def cancel(self):
        self.cancelled = True
    
    def get_finished_percentage(self) -> float:
        if self.total_scans == 0:
            return 1.0
        return self.parsed_scans / self.total_scans
    
    def _encode_array(self, data: np.ndarray, precision: str = "64") -> str:
        """Encode array data to base64 with optional compression"""
        # Convert to appropriate byte format
        if precision == "64":
            byte_data = data.astype(np.float64).tobytes()
        else:
            byte_data = data.astype(np.float32).tobytes()
        
        # Apply compression
        if self.compression == CompressionType.ZLIB:
            byte_data = zlib.compress(byte_data)
        elif self.compression == CompressionType.GZIP:
            byte_data = gzip.compress(byte_data)
        
        # Base64 encode
        return base64.b64encode(byte_data).decode('ascii')
    
    def _create_cv_param_element(self, xml: XMLBuffer, cv_param: CVParam):
        """Create a cvParam XML element"""
        elem = xml.tag("cvParam")
        elem.set("cvRef", "MS")
        elem.set("accession", cv_param.accession)
        elem.set("name", cv_param.name)
        elem.set("value", cv_param.value)
        
        if cv_param.unit_accession:
            elem.set("unitAccession", cv_param.unit_accession)
        
        elem.write()
        
        return elem
    
    def export(self) -> bool:
        """Export the raw data file to mzML format"""

        xml = XMLBuffer()

        try:
            self.logger.info(f"Started export of {self.raw_data_file.file_name} to {self.output_path}")
            
            # Create root element
            with (xml.tag("indexedmzML")
                .set("xsi:schemeLocation", "http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd")
                .set("xmlns", "http://psi.hupo.org/ms/mzml")
                .set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")) as root:
                # mzML element
                with (xml.tag("mzML")
                    .set("xsi:schemeLocation", "http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd")
                    .set("id", self.raw_data_file.file_name)
                    .set("version", "1.1.0")
                    .set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")) as mzml:
                    # cvList (empty for now)
                    with (xml.tag("cvList")
                        .set("count", "1")) as cv_list:

                        cv = xml.tag("cv")
                        cv.set("id", "MS")
                        cv.set("fullName", "Proteomics Standards Initiative Mass Spectrometry Ontology")
                        cv.set("version", "4.1.30")
                        cv.set("URI", "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo")
                        cv.write()

                    # dataProcessingList
                    with (xml.tag("dataProcessingList")
                          .set("count", "1")) as data_proc_list:
                        with (xml.tag("dataProcessing")
                            .set("id", "NETCDF_mzml_export")) as data_proc:

                            proc_method = xml.tag("processingMethod")
                            proc_method.set("softwareRef", "NETCDF_converter")
                            proc_method.set("order", "0")
                            proc_method.write()

                    # run
                    with (xml.tag("run")
                          .set("id", self.raw_data_file.file_name)
                          .set("startTimeStamp", self.raw_data_file.run_date.strftime("%Y-%m-%dT%H:%M:%SZ") if self.raw_data_file.run_date else "")
                          .set("defaultInstrumentConfigurationRef", "unknown")):
                        # spectrumList
                        with (xml.tag("spectrumList")
                              .set("count", str(len(self.raw_data_file.scans)))
                              .set("defaultDataProcessingRef", "NETCDF_mzml_export")):

                            spectrum_offsets = []

                            # Process each scan
                            for i, scan in enumerate(self.raw_data_file.scans):
                                if self.cancelled:
                                    return False

                                # Track offset for indexing (approximate - would need byte-level tracking for exact)
                                spectrum_offsets.append(xml.get_offset())  # Placeholder offset

                                # spectrum element
                                with (xml.tag("spectrum")
                                      .set("index", str(i))
                                      .set("id", f"scan={scan.scan_number}")
                                      .set("defaultArrayLength", str(len(scan.mz_values)))):

                                    # Spectrum type CV param
                                    if scan.spectrum_type == MassSpectrumType.CENTROIDED:
                                        self._create_cv_param_element(xml, self.CV_PARAMS['centroided'])
                                    else:
                                        self._create_cv_param_element(xml, self.CV_PARAMS['profile'])

                                    # MS level CV param
                                    ms_level_param = CVParam(self.CV_PARAMS['ms_level'].accession, self.CV_PARAMS['ms_level'].name, str(scan.ms_level))
                                    self._create_cv_param_element(xml, ms_level_param)

                                    # TIC CV param
                                    if scan.tic is not None:
                                        tic_param = CVParam(self.CV_PARAMS['tic'].accession, self.CV_PARAMS['tic'].name, str(scan.tic))
                                        self._create_cv_param_element(xml, tic_param)

                                    # m/z range CV params
                                    if scan.mz_range:
                                        low_mz_param = CVParam(self.CV_PARAMS['lowest_mz'].accession, self.CV_PARAMS['lowest_mz'].name, str(scan.mz_range[0]), self.CV_PARAMS['lowest_mz'].unit_accession)
                                        self._create_cv_param_element(xml, low_mz_param)

                                        high_mz_param = CVParam(self.CV_PARAMS['highest_mz'].accession, self.CV_PARAMS['highest_mz'].name, str(scan.mz_range[1]), self.CV_PARAMS['highest_mz'].unit_accession)
                                        self._create_cv_param_element(xml, high_mz_param)

                                    # scanList
                                    with xml.tag("scanList").set("count", "1"):
                                        with xml.tag("scan"):
                                            # Retention time CV param
                                            rt_param = CVParam(self.CV_PARAMS['scan_start_time'].accession, self.CV_PARAMS['scan_start_time'].name, str(scan.retention_time * 60), self.CV_PARAMS['scan_start_time'].unit_accession)
                                            self._create_cv_param_element(xml, rt_param)

                                            # Polarity CV param
                                            if scan.polarity == PolarityType.POSITIVE:
                                                self._create_cv_param_element(xml, self.CV_PARAMS['positive_polarity'])
                                            elif scan.polarity == PolarityType.NEGATIVE:
                                                self._create_cv_param_element(xml, self.CV_PARAMS['negative_polarity'])

                                            # Filter string CV param
                                            filter_param = CVParam(self.CV_PARAMS['filter_string'].accession, self.CV_PARAMS['filter_string'].name, scan.scan_definition)
                                            self._create_cv_param_element(xml, filter_param)

                                            # scanWindowList
                                            with xml.tag("scanWindowList").set("count", "1"):
                                                with xml.tag("scanWindow"):
                                                    # Scan window range (if available)
                                                    if scan.mz_range:
                                                        lower_param = CVParam(self.CV_PARAMS['scan_window_lower'].accession, self.CV_PARAMS['scan_window_lower'].name, str(scan.mz_range[0]), self.CV_PARAMS['scan_window_lower'].unit_accession)
                                                        self._create_cv_param_element(xml, lower_param)

                                                        upper_param = CVParam(self.CV_PARAMS['scan_window_upper'].accession, self.CV_PARAMS['scan_window_upper'].name, str(scan.mz_range[1]), self.CV_PARAMS['scan_window_upper'].unit_accession)
                                                        self._create_cv_param_element(xml, upper_param)

                                            # binaryDataArrayList
                                    with xml.tag("binaryDataArrayList").set("count", "2"):
                                        # m/z array
                                        mz_encoded = self._encode_array(scan.mz_values, "64")
                                        compression_param = CVParam(self.compression.accession, self.compression._name)
                                        with xml.tag("binaryDataArray").set("encodedLength", str(len(mz_encoded))):
                                            self._create_cv_param_element(xml, self.CV_PARAMS['64_bit_float'])
                                            self._create_cv_param_element(xml, compression_param)
                                            self._create_cv_param_element(xml, CVParam(self.CV_PARAMS['mz_array'].accession, self.CV_PARAMS['mz_array'].name, unit_accession=self.CV_PARAMS['mz_array'].unit_accession))
                                            xml.tag("binary").write(mz_encoded)

                                        # intensity array
                                        intensity_encoded = self._encode_array(scan.intensity_values, "32")
                                        with xml.tag("binaryDataArray").set("encodedLength", str(len(intensity_encoded))):
                                            self._create_cv_param_element(xml, self.CV_PARAMS['32_bit_float'])
                                            self._create_cv_param_element(xml, compression_param)
                                            self._create_cv_param_element(xml, CVParam(self.CV_PARAMS['intensity_array'].accession, self.CV_PARAMS['intensity_array'].name, unit_accession=self.CV_PARAMS['intensity_array'].unit_accession))
                                            xml.tag("binary").write(intensity_encoded)

                                self.parsed_scans += 1
            
                        # chromatogramList (empty for NetCDF conversion)
                        chrom_list = xml.tag("chromatogramList")
                        chrom_list.set("count", "0")
                        chrom_list.set("defaultDataProcessingRef", "NETCDF_mzml_export")
                        chrom_list.write()

                list_offset = xml.get_offset()
                # indexList
                with xml.tag("indexList").set("count", "1"):
                    # spectrum index
                    with xml.tag("index").set("name", "spectrum"):

                        for i, scan in enumerate(self.raw_data_file.scans):
                            offset_elem = xml.tag("offset")
                            offset_elem.set("idRef", f"scan={scan.scan_number}")
                            offset_elem.write(str(spectrum_offsets[i]))

                # indexListOffset
                index_offset = xml.tag("indexListOffset")
                index_offset.write(str(list_offset))

                # fileChecksum (simplified)
                file_checksum = xml.tag("fileChecksum")
                file_checksum.write("0" * 40)  # Placeholder SHA1

            # Write to file
            self._write_xml_file(xml)
            
            self.logger.info(f"Finished export of {self.raw_data_file.file_name}, exported {self.parsed_scans} scans")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export file: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _write_xml_file(self, xml: XMLBuffer):
        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        """Write XML element tree to file with proper formatting"""
        with open(self.output_path, 'wb+') as f:
            f.write(xml.buffer)


class NetCDFImportTask:
    """
    Python equivalent of the Java NetCDFImportTask class.
    Parses mass spectrometry data from NetCDF files.
    """
    
    def __init__(self, file_path: Path):
        self.logger = logging.getLogger(__name__)
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        
        # Task status tracking
        self.status = TaskStatus.PROCESSING
        self.error_message: str | None = None
        self.parsed_scans = 0
        self.total_scans = 0
        self.number_of_good_scans = 0
        self.scan_num = 0
        
        # NetCDF file and variables
        self.input_file: nc.Dataset | None = None
        self.mass_value_variable = None
        self.intensity_value_variable = None
        
        # Scale factors (some software uses scale factors like 0.05)
        self.mass_value_scale_factor = 1.0
        self.intensity_value_scale_factor = 1.0
        
        # Scan metadata
        self.scans_index: dict[int, tuple[int, int]] = {}  # scan_num -> (start_pos, length)
        self.scans_retention_times: dict[int, float] = {}
        
        # Result container
        self.raw_data_file = RawDataFile(self.file_name, file_path)
    
    def get_finished_percentage(self) -> float:
        """Returns the percentage of completion (0.0 to 1.0)"""
        if self.total_scans == 0:
            return 0.0
        return self.parsed_scans / self.total_scans
    
    def get_task_description(self) -> str:
        return f"Opening file {self.file_path}"
    
    def is_cancelled(self) -> bool:
        """Override this method to implement cancellation logic"""
        return False
    
    def run(self) -> RawDataFile | None:
        """
        Main execution method that parses the NetCDF file.
        Returns the parsed RawDataFile or None if an error occurred.
        """
        self.status = TaskStatus.PROCESSING
        self.logger.info(f"Started parsing file {self.file_path}")
        
        try:
            # Open and parse the file
            self.start_reading()
            
            # Parse scans
            while True:
                if self.is_cancelled():
                    self.status = TaskStatus.CANCELLED
                    return None
                
                building_scan = self.read_next_scan()
                if building_scan is None:
                    break
                
                self.raw_data_file.add_scan(building_scan)
                self.parsed_scans += 1
            
            # Finish reading
            self.finish_reading()
            
            self.logger.info(f"Finished parsing {self.file_path}, parsed {self.parsed_scans} scans")
            self.status = TaskStatus.FINISHED
            return self.raw_data_file
            
        except Exception as e:
            self.logger.error(f"Could not open file {self.file_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.error_message = str(e)
            self.status = TaskStatus.ERROR
            return None
    
    def start_reading(self):
        """Initialize NetCDF file reading and extract metadata"""
        try:
            # Open NetCDF file
            self.input_file = nc.Dataset(self.file_path, 'r')
        except Exception as e:
            raise IOError(f"Couldn't open input file {self.file_path}: {str(e)}")
        
        if 'netcdf_file_date_time_stamp' in self.input_file.ncattrs():
            self.raw_data_file.run_date = datetime.strptime(self.input_file.netcdf_file_date_time_stamp, "%Y%m%d%H%M%S%z")
        else:
            logging.warning("No run date found in NetCDF file")

        # Find mass_values and intensity_values variables
        if 'mass_values' not in self.input_file.variables:
            raise IOError("Could not find variable mass_values")
        self.mass_value_variable = self.input_file.variables['mass_values']
        
        if 'intensity_values' not in self.input_file.variables:
            raise IOError("Could not find variable intensity_values")
        self.intensity_value_variable = self.input_file.variables['intensity_values']
        
        # Check for scale factors
        if hasattr(self.mass_value_variable, 'scale_factor'):
            self.mass_value_scale_factor = float(self.mass_value_variable.scale_factor)
        
        if hasattr(self.intensity_value_variable, 'scale_factor'):
            self.intensity_value_scale_factor = float(self.intensity_value_variable.scale_factor)
        
        # Read scan index information
        if 'scan_index' not in self.input_file.variables:
            raise IOError(f"Could not find variable scan_index from file {self.file_path}")
        
        scan_index_variable = self.input_file.variables['scan_index']
        self.total_scans = scan_index_variable.shape[0]
        
        # Read scan start positions
        scan_start_positions = np.zeros(self.total_scans + 1, dtype=int)
        scan_start_positions[:-1] = scan_index_variable[:]
        scan_start_positions[-1] = self.mass_value_variable.size  # End position for last scan
        
        # Read retention times
        if 'scan_acquisition_time' not in self.input_file.variables:
            raise IOError(f"Could not find variable scan_acquisition_time from file {self.file_path}")
        
        scan_time_variable = self.input_file.variables['scan_acquisition_time']
        retention_times = scan_time_variable[:] / 60.0  # Convert to minutes
        
        # Handle missing scans (QStar data converter fix)
        self._fix_missing_scans(scan_start_positions, retention_times)
        
        # Store scan metadata
        for i in range(self.total_scans):
            start_pos = scan_start_positions[i]
            length = scan_start_positions[i + 1] - scan_start_positions[i]
            
            self.scans_index[i] = (start_pos, length)
            self.scans_retention_times[i] = float(retention_times[i])
    
    def _fix_missing_scans(self, scan_start_positions: np.ndarray, retention_times: np.ndarray):
        """Fix problems caused by missing scans in QStar data"""
        # Count good scans
        self.number_of_good_scans = np.sum(scan_start_positions[:-1] >= 0)
        
        if self.number_of_good_scans < self.total_scans:
            # Calculate average time delta between scans
            deltas = []
            for i in range(self.total_scans):
                if scan_start_positions[i] >= 0:
                    for j in range(i + 1, self.total_scans):
                        if scan_start_positions[j] >= 0:
                            delta = (retention_times[j] - retention_times[i]) / (j - i)
                            deltas.append(delta)
                            break
            
            avg_delta = np.mean(deltas) if deltas else 0.0
            
            # Fix missing retention times
            for i in range(self.total_scans):
                if scan_start_positions[i] < 0:
                    # Find nearest good scan
                    nearest_idx = None
                    min_distance = float('inf')
                    
                    for j in range(self.total_scans):
                        if scan_start_positions[j] >= 0:
                            distance = abs(j - i)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_idx = j
                    
                    if nearest_idx is not None:
                        retention_times[i] = retention_times[nearest_idx] + (i - nearest_idx) * avg_delta
                    else:
                        retention_times[i] = retention_times[i-1] if i > 0 else 0.0
            
            # Fix scan start positions
            for i in range(self.total_scans):
                if scan_start_positions[i] < 0:
                    for j in range(i + 1, self.total_scans + 1):
                        if scan_start_positions[j] >= 0:
                            scan_start_positions[i] = scan_start_positions[j]
                            break
    
    def read_next_scan(self) -> Scan | None:
        """Read the next scan from the NetCDF file"""
        # Check if we've reached the end
        if self.scan_num not in self.scans_index:
            return None

        definition = ""

        if "global_mass_min" in self.input_file.ncattrs() and "global_mass_max" in self.input_file.ncattrs():
            mz_min = float(self.input_file.getncattr("global_mass_min"))
            mz_max = float(self.input_file.getncattr("global_mass_max"))
            definition = f"FTMS + p ESI w SIM ms [{mz_min:.02f}-{mz_max:.02f}]"
        
        start_pos, length = self.scans_index[self.scan_num]
        retention_time = self.scans_retention_times[self.scan_num]
        
        # Handle empty scans
        if length == 0:
            self.scan_num += 1
            return Scan(
                file_name=self.file_name,
                scan_number=self.scan_num,
                ms_level=1,
                retention_time=retention_time,
                mz_values=np.array([]),
                intensity_values=np.array([]),
                spectrum_type=MassSpectrumType.CENTROIDED,
                polarity=PolarityType.UNKNOWN,
                scan_definition=definition
            )
        
        # Read mass and intensity values
        try:
            mz_values = self.mass_value_variable[start_pos:start_pos + length] * self.mass_value_scale_factor
            intensity_values = self.intensity_value_variable[start_pos:start_pos + length] * self.intensity_value_scale_factor
        except Exception as e:
            raise IOError("Could not read from variables mass_values and/or intensity_values") from e
        
        # Convert to numpy arrays
        mz_values = np.asarray(mz_values, dtype=np.float64)
        intensity_values = np.asarray(intensity_values, dtype=np.float64)
        
        # Auto-detect spectrum type
        spectrum_type = detect_spectrum_type(mz_values, intensity_values)
        
        self.scan_num += 1

        return Scan(
            file_name=self.file_name,
            scan_number=self.scan_num,
            ms_level=1,
            retention_time=retention_time,
            mz_values=mz_values,
            intensity_values=intensity_values,
            spectrum_type=spectrum_type,
            polarity=PolarityType.UNKNOWN,  # NetCDF doesn't typically contain polarity info
            scan_definition=definition
        )

    
    def finish_reading(self):
        """Close the NetCDF file"""
        if self.input_file:
            self.input_file.close()
            self.input_file = None

    def get_imported_raw_data_file(self) -> RawDataFile | None:
        """Return the imported data file if parsing was successful"""
        return self.raw_data_file if self.status == TaskStatus.FINISHED else None


def convert_netcdf_file(file_path: Path, dst_path: Path):
    parser = NetCDFImportTask(file_path)
    raw_data = parser.run()
    if raw_data is None:
        print(f"Error parsing file: {parser.error_message}")
        return
    exporter = MzMLExporter(raw_data, dst_path, CompressionType.ZLIB)
    success = exporter.export()
    if success:
        print("Exported to output.mzML successfully")
    else:
        print("Failed to export to mzML", )


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Convert one or more files/directories from NetCDF to mzML format"
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",  # one or more paths
        help="One or more input files or directories"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional output file or directory"
    )

    args = parser.parse_args()

    if args.output and not args.output.exists():
        try:
            args.output.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            parser.error(f"Could not create output directory: {args.output}, {str(e)}")

    files: dict[Path, Path] = {}

    # Validate input paths
    for p in args.paths:
        if not p.exists():
            parser.error(f"Input path does not exist: {p}")
        elif p.is_file():
            files[p] = (args.output.resolve() if args.output else p.parent) / (p.name.rsplit('.', maxsplit=1)[0] + '.mzML')
        elif p.is_dir():
            # Add all files in the directory
            dir_files = {f: (args.output.resolve() / p.name if args.output else p) / (f.name.rsplit('.', maxsplit=1)[0] + '.mzML') for f in p.iterdir() if f.is_file() and (f.name.endswith(".nc") or f.name.endswith(".cdf"))}
            if not dir_files:
                parser.error(f"No files found in directory: {p}")
            else:
                files.update(dir_files)

    for file_path, dst_path in files.items():
        try:
            print(f"Converting {file_path}")
            convert_netcdf_file(file_path, dst_path)
        except Exception as e:
            print(f"Error converting file {file_path}: {str(e)}")
            print(traceback.format_exc())



# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    main()

    

