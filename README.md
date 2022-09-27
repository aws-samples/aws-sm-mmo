## Serve multiple models with One Sagamaker Endpoint 

1.	Launch a g4dn.2xlarge instance with Deep Learning AMI (Ubuntu 20.04), adjust EBS storage size to at least 50G.
2.	Clone this repo
```
git clone https://github.com/aws-samples/aws-sm-mmo
```
3. Extract model file
```
cd model/bert_base/1/
unzip model.bin.zip
```
4. Install and config aws-cli, and git.
```
sudo apt install awscli
sudo apt install git-all
```
config aws client
```
aws configure
# input AWS Access Key ID, AWS Secret Access Key, Default region name and Default output format
```
5. Build docker image
```
./build.sh mytriton
```
6. Launch a notebook instance with ml.c5.xlarge specification from SageMaker console
7. Upload mytriton.ipynb to notebook instance, then execute step by step.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

