module Utils where

import Types

aproxEq :: Double -> Double -> Bool
aproxEq a b = abs (a - b) < 1e-6

mse :: Targets -> Results -> Double
mse ts rs = (0.5) * (sum . map (** 2) . zipWith (\t r -> r - t) ts) rs