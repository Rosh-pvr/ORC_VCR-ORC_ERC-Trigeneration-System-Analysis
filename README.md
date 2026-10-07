# ORC_VCR
**Circuit for ORC (Organic Rankine cycle)_VCR(Vapor compression refrigeration)** :
[![ORC-ERC.jpg](https://i.postimg.cc/4ysPvLMv/ORC-ERC.jpg)](https://postimg.cc/LYQj2Tmh)

**Circuit for Circuit for ORC (Organic Rankine cycle)_ERC(Ejector Refrigeration Cycle)**:
[![ORC-ERC.jpg](https://i.postimg.cc/4ysPvLMv/ORC-ERC.jpg)](https://postimg.cc/LYQj2Tmh)

**Recommended Computing Variables**:

|  |  | No. | Column | Symbol | Unit | Status |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | 1 | Fuel | — | — | Existing |
|  |  | 2 | Configuration | ORC / ORC-VCR | — | Add |
|  |  | 3 | Compression Ratio | CR | — | Existing |
|  |  | 4 | Engine Load | Load | % | Existing |
|  |  | 5 | Exhaust Gas Temperature | T21 | °C | Existing |
|  |  | 6 | Exhaust Gas Mass Flow | m21 | kg/s | Existing |
|  |  | 7 | Thermal Oil Mass Flow | m14 / m_oil | kg/s | Existing |
|  |  | 8 | ORC Working-Fluid Mass Flow | mORC | kg/s | ADD |
|  |  | 9 | ORC Net Power | WnetORC | kW | COMPUTE |
|  |  | 10 | VCR COP | COPVCR | — | COMPUTE |
|  |  | 11 | ORC Exergy Efficiency | ηex,ORC | % | COMPUTE |
|  |  | 12 | System Exergy Efficiency | ηex,sys | % | COMPUTE / recommended |

