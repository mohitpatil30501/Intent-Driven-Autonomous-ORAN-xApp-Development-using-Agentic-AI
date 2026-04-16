# O-RAN Test Results JSON Schema

## Introduction

This project provides the JSON schemas required to implement the Test Results Artifact, defined as part of the O-RAN Certification and Badging Processes and Procedures.  

In general, each schema also provides one or two examples of objects meeting the adhering of the schema. There may also be equalivelent examples in `yaml`.  To convert `json` to `yaml`, the follow bash command can be used, `cat FILENAME.json | yq -P`.  The convert `yaml` to `json`, the following bash command can be used, `cat FILENAME.yaml | jq`.

To validate the examples against the schema, tools such as [check-jsonschema](https://pypi.org/project/check-jsonschema/) can be used, such as the following `check-jsonschema --schemafile o-ran-test-results-schema.json examples/example-1.json`.  Tho aid in development, these tools can also check the schema file itself, `check-jsonschema --check-metaschema o-ran-test-results-schema.json`. Note, these schemas are based on the 2020-12 specification, and that version must be supported by the validation tools.

## Schemas

The following schemas have currently been defined.

### o-ran-test-results schema

This schema contains the test results, along with information pertaining to the testbed, test lab, etc.  The schema implements the following structure.  Note, the schema will enforce data types, minimum array sides, and object parameter names.

```
json schema
├── $schema
├── Schema Version
├── testMetadata
│  ├── Start Date
│  ├── End Date
│  ├── [Contacts] - Array of objects
│      ├── firstName
│      ├── lastName
│      ├── Email
│      ├── Phone
│  ├── DUT Name
│  ├── Interface Under Test - Array of enums
│  ├── Result - enum
│  ├── testType - enum
│  ├── testID
│  ├── iotProfile - subschema object
├── Testbed Components
│  ├── [Component] - array of objects
│    ├── Description
│    ├── Vendor Name
│    ├── Model Number
│    ├── Software Version
│    ├── Firmware Version
│    ├── Hardware Version
│    ├── [Contacts] - Array of objects (same format as above)
│    ├── [configurationArtifacts] - Array of objects
│      ├── Name
│      ├── Path (local)
│      ├── Description
├── testLab
│  ├── Name
│  ├── Address
│  ├── [Contacts] - Array of objects (same format as above)
│  ├── [Links] - Array of objects
│    ├── displayName
│    ├── url
│    ├── Description
├── testSpecification
│  ├── Name
│  ├── Version
│  ├── Description
│  ├── [Links] - Array of objects (same format as above)
├── [Tags] - Array of strings
├── [Notes] - Array of objects
│  ├── Title
│  ├── Body
├── testResults
│  ├── [Test Group or Test Case] - Array of either subschema object



Test Group
├── Number
├── Name
├── Description
├── [Test Group or Test Case] - Array of either subschema object



Test Case
├── Number
├── Name
├── Description
├── Status - enum
├── Result - enum
├── [Artifacts] - Array of objects
│  ├── Name
│  ├── Path (local)
│  ├── Description
├── [ Links] - Array of objects (same format as above)
├── [Measurements] - Array of objects
│  ├── Name
│  ├── Description
│  ├── [Values] - Array of values. All elements should be in the same unit.
│  ├── Units - enum
├── [Metrics] - Array of objects
│  ├── Description
│  ├── Status - enum
│  ├── Result - enum
│  ├── [Measurements] - Array of objects (same format as above)
├── [Notes] - Array of objects (same format as above)
├── Start Date
├── Stop Date
├── [Contacts] - Array of objects (same format as above)
```

### o-ran-wg4-iot-profile schema

This schema provides details about the WG4 IOT profile utilized during the testing

```
iotProfile
├── WG4 IOT Spec Version
├── M-Plane IOT Profile Name
├── M-Plane IOT Profile Test Configuration Name
├── CUS-Plane IOT Profile Name
├── CUS-Plane IOT Profile Test Configuration Name
```

### o-ran-config-parameters schema

This schema provides various configuration parameters that may be applied during testing. Within the object, all specific properties are optional.  Where possible, the specific properties use enums, number requirements, or other formating requirements, to enforce the correct configuration parameter types, ranges, or formats.

```
configurationParameters
├── deploymentArchitecture - enum
├── deploymentScale - enum
├── deploymentRfScenario - enum
├── frequencyRange5G - Array of enums
├── band5G - Array of enums
├── bandLTE - Array of enums
├── nr-arfcn - Positive float
├── e-arfcn - Postive float
├── subCarrierSpacing - enum
├── totalTransmissionBandwidth - Postive float
├── totalResourceBlocks - Postive integer
├── carrierPrefixLength - Postive integer
├── slotLength - Postive integer
├── duplexMode - enum
├── tddDlUlRatio - Postive float
├── ipv4 - boolean
├── ipv6 - boolean
├── numMimoLayers - Postive integer
├── numTxAntenna - Postive integer
├── numRxAntenna - Postive integer
├── totalAntennaGain - float
├── totalTransmitPowerIntoAntenna - float
```

#### Revision Hostory:

* v1 - Initial release of the schema
