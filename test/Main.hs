module Main (main) where

import MyLib (someFunc)

import Test.HUnit

testBase :: Test
testBase = TestCase (assertEqual "Some Func" "Função" (someFunc))

allTests :: Test
allTests = TestList [TestLabel "Test Base" testBase]

main :: IO ()
main = do
  putStrLn "Running tests..."
  runTestTT allTests
  return ()
