module MyLib where

import Layers
import Model

generateModel :: [MetaLayer] -> IO (Model)
generateModel metaLayers = Model <$> mapM genLayer' metaLayers