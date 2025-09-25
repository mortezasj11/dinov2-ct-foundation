

1. Run the following command to create the headless service:
# kubectl apply -f headless-service.yaml
# kubectl apply --kubeconfig=/rsrch1/ip/msalehjahromi/.kube/advanced-config.json -f headless-service.yaml

This creates a headless service named msalehjahromi-torchrun-test

2. Deploy the Multi-Node Job
Once the headless service is up, deploy your multi-node Job. Save your updated Job YAML (from above) to a file, e.g., torchrun-job.yaml, and apply it with:

# kubectl apply -f multi_node_job_service_first.yaml
# kubectl apply --kubeconfig=/rsrch1/ip/msalehjahromi/.kube/advanced-config.json -f multi_node_job_service_first.yaml

3. Verify the Setup
Check that:

Headless Service: Ensure the service exists and has no clusterIP:

bash
Copy code
kubectl get svc msalehjahromi-torchrun-dl5-b8-2 -n yn-gpu-workload
Output should show:

scss
Copy code
NAME                                TYPE        CLUSTER-IP   PORT(S)    AGE
msalehjahromi-torchrun-dl5-b8-2     ClusterIP   None         <none>     XXm
Pods: Verify that both pods are running:

bash
Copy code
kubectl get pods -n yn-gpu-workload
You should see two pods with names like:

sql
Copy code
msalehjahromi-torchrun-dl5-b8-2-0   Running   ...
msalehjahromi-torchrun-dl5-b8-2-1   Running   ...
DNS Resolution: Test that pods can resolve each other’s hostnames using DNS.

From within Pod-1 (rank 1), try resolving the DNS name for Pod-0 (rank 0):
bash
Copy code
kubectl exec -it <POD-NAME> -n yn-gpu-workload -- nslookup msalehjahromi-torchrun-dl5-b8-2-0.msalehjahromi-torchrun-dl5-b8-2.yn-gpu-workload.svc.cluster.local
This should return the IP of Pod-0.
4. Logs and Debugging
Check logs to ensure all nodes have successfully connected and training has started:
bash
Copy code
kubectl logs msalehjahromi-torchrun-dl5-b8-2-0 -n yn-gpu-workload
kubectl logs msalehjahromi-torchrun-dl5-b8-2-1 -n yn-gpu-workload
5. Common Issues and Solutions
Pods Cannot Resolve Hostnames:

Verify the headless service is working with kubectl describe svc msalehjahromi-torchrun-dl5-b8-2.
Ensure the k8s-user: msalehjahromi label is on both pods.
Node Rank Mismatch:

Confirm that NODE_RANK is correctly set using kubectl describe pod <pod-name> and checking the environment variables.
Master Node Not Ready:

Pod-1 might attempt to connect to Pod-0 before it’s ready. This typically resolves itself as PyTorch retries connections. If not, consider a StatefulSet for more robust pod startup handling.
In Summary
Create the Headless Service first.
Deploy the Job after the service is running.
Verify pods are running and can resolve each other via DNS.
Check logs to ensure multi-node communication is working.
If you encounter any issues or need adjustments, let me know!