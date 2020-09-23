include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)

PrepareTest(VERONAC_FLAGS EXPECTED_DUMP ACTUAL_DUMP)

CheckStatus(
  COMMAND ${MLIRGEN} ${TEST_FILE}
  -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}-mlir.txt
  --pass-pipeline=typechecker,print-facts
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}-facts.txt
  EXPECTED_STATUS 0)

FileCheck(${TEST_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}-facts.txt)
