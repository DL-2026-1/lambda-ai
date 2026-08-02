module Main (main) where

import Test.HUnit

import IrisTest

allTests :: Test
allTests = TestList [testIris]

main :: IO ()
main = do
  putStrLn "Running tests..."
  runTestTT allTests
  return ()
