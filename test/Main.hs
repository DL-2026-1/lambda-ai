module Main (main) where

import Test.HUnit

allTests :: Test
allTests = TestList []

main :: IO ()
main = do
  putStrLn "Running tests..."
  runTestTT allTests
  return ()
