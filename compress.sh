cd ..
rm -rf code.tar.gz;
tar -zc --exclude='*.git' --exclude='*.pyth' --exclude='.vscode' --exclude='output' --exclude='current_epoch' -f code.tar.gz Masked-Action-Recognition requirements.txt
