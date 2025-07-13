# DGA Database - Dissolved Gas Analysis for Transformer Fault Diagnosis

A comprehensive, publicly accessible database of transformer dissolved gas analysis (DGA) data for research and educational purposes.

[![License: Open Access](https://img.shields.io/badge/License-Open%20Access-green.svg)](https://github.com/bmrayan/dgadb)
[![Dataset Size](https://img.shields.io/badge/Dataset-743%20Transformers-blue.svg)](https://bmrayan.github.io/dgadb)
[![Gas Parameters](https://img.shields.io/badge/Parameters-5%20Gases-orange.svg)](https://bmrayan.github.io/dgadb)

## 🎯 Overview

This repository contains a quality-controlled dataset of dissolved gas analysis measurements from 743 transformers, designed to support research in transformer condition monitoring and fault diagnosis. The database addresses the critical challenge of limited public access to high-quality DGA data required for developing and validating diagnostic algorithms.

## 📊 Dataset Characteristics

- **Total Records**: 743 transformer measurements
- **Gas Parameters**: 5 key dissolved gases (H₂, CH₄, C₂H₄, C₂H₆, C₂H₂)
- **Fault Cases**: 18 labeled fault conditions
- **Data Quality**: Systematic cleaning and validation
- **Coverage**: Diverse geographical and operational conditions

### Statistical Summary

| Gas | Mean (ppm) | Max (ppm) | Skewness | Range |
|-----|------------|-----------|----------|-------|
| H₂ | 348.4 | 23,349 | 7.76 | 4 orders |
| CH₄ | 160.2 | 11,646 | 9.63 | 4 orders |
| C₂H₄ | 282.0 | 46,976 | 16.94 | 4 orders |
| C₂H₆ | 107.2 | 9,901 | 14.39 | 4 orders |
| C₂H₂ | 93.7 | 9,740 | 9.98 | 4 orders |

## 🌐 Web Interface

Access the interactive database at: **[https://bmrayan.github.io/dgadb/](https://bmrayan.github.io/dgadb/)**

### Features

- **Interactive Queries**: Filter data using simple expressions
- **Quick Access**: Predefined queries for common scenarios
- **Multiple Downloads**: CSV and JSON export options
- **Statistical Analysis**: Comprehensive visualizations and correlations
- **Methodology Documentation**: Complete data collection procedures

### Query Examples

```
Hydrogen > 200                    # High hydrogen levels
Fault == "Electrical"            # Electrical fault cases
Hydrogen > 100 AND Methane > 30   # Multiple conditions
Ethylene > 50 OR Acetylene > 5    # Alternative conditions
```

## 📁 Repository Structure

```
DGA_Dataset_Repo/
├── index.html              # Main database interface
├── analysis.html           # Statistical analysis and visualizations
├── collection.html         # Data collection methodology
├── StatDGA.py              # Data analysis and figure generation script
├── FinalDataSet_DGA.xlsx   # Raw dataset (Excel format)
├── web_data.json           # Web-formatted dataset
├── FinalFigures1/          # Generated statistical figures
│   ├── 01c2_*_distribution_log_only.png
│   ├── 02_correlation_analysis.png
│   ├── 03c_distribution_analysis_log_only.png
│   ├── 04c_*_histogram_log_y_axis.png
│   ├── 04_5_distribution_analysis_lognormal.png
│   └── 05_fault_analysis.png
└── README.md               # This documentation
```

## 🔧 Usage

### Web Interface Access

1. Visit [https://bmrayan.github.io/dgadb/](https://bmrayan.github.io/dgadb/)
2. Use the query interface to filter data
3. Download results in CSV or JSON format
4. Explore statistical analysis and methodology sections

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/bmrayan/dgadb.git
   cd dgadb
   ```

2. Set up a local web server:
   ```bash
   python -m http.server 8000
   # or
   npx serve .
   ```

3. Open `http://localhost:8000` in your browser

### Data Analysis

Run the statistical analysis script:

```bash
python StatDGA.py
```

**Requirements:**
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

## 📈 Statistical Analysis

The dataset includes comprehensive statistical characterization:

### Key Findings

- **Distribution Patterns**: All gases exhibit log-normal characteristics with high positive skewness (7.76-16.94)
- **Concentration Ranges**: Four orders of magnitude variation across all gas types
- **Inter-Gas Correlations**: Strong relationships among hydrocarbon gases (0.65-0.85)
- **Fault Signatures**: Distinct patterns for electrical vs. thermal fault conditions

### Generated Figures

1. **Gas Distributions**: Individual concentration patterns for each gas
2. **Correlation Analysis**: Inter-gas relationship heatmap
3. **Box Plots**: Statistical summaries on logarithmic scales
4. **Frequency Distributions**: Histogram analysis with logarithmic scaling
5. **Distribution Fitting**: Statistical model fitting and parameters
6. **Fault Analysis**: Gas signatures by fault type

## 🔬 Research Applications

### Supported Use Cases

- **Machine Learning**: Training and validation of diagnostic algorithms
- **Benchmarking**: Standardized comparison of diagnostic methods
- **Education**: Real-world data for teaching DGA interpretation
- **Method Validation**: Testing traditional ratio-based approaches
- **Algorithm Development**: Pattern recognition and classification research

### DGA Methods Supported

- Key Gas Method
- Rogers Ratio Method
- Doernenburg Ratio Method
- IEC Ratio Method (IEC 60599)
- Duval Triangle/Pentagon
- Machine Learning Approaches

## 📊 Data Quality

### Quality Assurance Process

1. **Source Verification**: Multi-source data validation
2. **Format Standardization**: Consistent units and naming
3. **Duplicate Detection**: Systematic duplicate removal
4. **Outlier Analysis**: Statistical validation of extreme values
5. **Integrity Verification**: Cross-validation against DGA patterns

### Quality Metrics

- **Initial Records**: 1,193 transformers
- **Final Dataset**: 743 transformers (62.3% retention)
- **Data Completeness**: 100% for all five gas parameters
- **Fault Labeling**: 18 verified fault cases

## 🚀 Future Development

### Planned Enhancements

- **Geographic Expansion**: Broader international representation
- **Fault Case Addition**: More labeled fault examples
- **Metadata Enrichment**: Transformer specifications and history
- **API Development**: Programmatic data access
- **Mobile Interface**: Native mobile application

### Community Contributions

We welcome contributions from the research and industry community:

- Additional transformer data
- Fault case validation
- Method implementations
- Documentation improvements
- Bug reports and feature requests

## 📚 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{dga_database_2024,
  title={DGA Database: Comprehensive Dissolved Gas Analysis Dataset for Transformer Fault Diagnosis},
  author={[Authors]},
  year={2024},
  url={https://bmrayan.github.io/dgadb/},
  note={Open access dataset for transformer condition monitoring research}
}
```

## 📄 License

This dataset is provided under an **Open Access License** for research and educational purposes.

### Terms of Use

- ✅ Research and academic use
- ✅ Educational purposes
- ✅ Non-commercial applications
- ✅ Method development and validation
- ❌ Commercial redistribution without permission
- ❌ Claims of ownership or exclusivity

## 🤝 Acknowledgments

We acknowledge the contributions of:

- Research institutions providing data
- Utility companies sharing operational data
- Testing laboratories for measurement validation
- Academic community for feedback and validation

## 📞 Contact

For questions, contributions, or collaboration:

- **Web Interface**: [https://bmrayan.github.io/dgadb/](https://bmrayan.github.io/dgadb/)
- **Repository**: [https://github.com/bmrayan/dgadb](https://github.com/bmrayan/dgadb)
- **Issues**: [Report bugs or request features](https://github.com/bmrayan/dgadb/issues)

## 🔄 Version History

### Current Version: 1.0.0

- Initial release with 743 transformer records
- Complete web interface with query capabilities
- Statistical analysis and visualization suite
- Comprehensive documentation and methodology

### Planned Updates

- v1.1.0: Additional fault cases and geographic expansion
- v1.2.0: API access and programmatic interface
- v2.0.0: Enhanced metadata and temporal data

---

**Disclaimer**: This dataset is provided for research purposes. Users should validate results against established standards and practices. The maintainers provide no warranty regarding data accuracy or fitness for specific applications.
