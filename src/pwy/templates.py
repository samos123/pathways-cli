YAML_TEMPLATE = """apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {NAME}
  namespace: {NAMESPACE}
spec:
  failurePolicy:
    maxRestarts: 0
    restartStrategy: BlockingRecreate
  replicatedJobs:
    # -------------------------------------------------------------------------
    # 1. Pathways Head (Client Pod)
    # -------------------------------------------------------------------------
    - name: pwhd
      replicas: 1
      template:
        spec:
          parallelism: 1
          completions: 1
          backoffLimit: 32
          template:
            metadata:
              annotations:
                cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
            spec:
              terminationGracePeriodSeconds: 60
              restartPolicy: Never
              hostAliases:
                - ip: 169.254.169.254
                  hostnames:
                    - metadata
                    - metadata.google.internal
{TOLERATIONS_HEAD}
{AFFINITY_HEAD}
              containers:
                - name: client
                  image: {CLIENT_IMAGE}
                  command:
                    - bash
                    - -c
                    - |
                      {CLIENT_EXECUTION_COMMAND}
                  resources:
                    requests:
                      cpu: "1000m"
                      memory: "16Gi"
                    limits:
                      cpu: "1000m"
                      memory: "16Gi"
                  env:
                    - name: TPU_TYPE
                      value: {TPU_TYPE}
                    - name: NUM_TPU_SLICES
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                    - name: JAX_BACKEND_TARGET
                      value: grpc://localhost:29000
                    - name: XCLOUD_ENVIRONMENT
                      value: GCP
                    - name: JAX_PLATFORMS
                      value: proxy
                    - name: ENABLE_PATHWAYS_PERSISTENCE
                      value: "1"
                    - name: TPU_SKIP_MDS_QUERY
                      value: "true"
                    - name: PYTHONUNBUFFERED
                      value: "1"
                    - name: TEST_UNDECLARED_OUTPUTS_DIR
                      value: "true"
                    - name: IFRT_PROXY_LARGE_TRANSFER_THRESHOLD
                      value: "1"
                    - name: IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY
                      value: /tmp/ifrt_proxy
                  volumeMounts:
                    - name: shared-memory
                      mountPath: /tmp/ifrt_proxy
                  imagePullPolicy: Always
              initContainers:
                - name: pathways-proxy
                  image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:jax-0.10.0
                  restartPolicy: Always
                  ports:
                    - containerPort: 29000
                  env:
                    - name: IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS
                      value: "true"
                    - name: IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY
                      value: /tmp/ifrt_proxy
                  args:
                    - --resource_manager_address=localhost:29001
                    - --server_port=29000{PROXY_SIDECAR_ARG}
                  volumeMounts:
                    - name: shared-memory
                      mountPath: /tmp/ifrt_proxy
                - name: pathways-rm
                  image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.10.0
                  restartPolicy: Always
                  env:
                    - name: TPU_SKIP_MDS_QUERY
                      value: "true"
                  args:
                    - --server_port=29001
                    - --node_type=resource_manager
                    - --instance_count={NUM_SLICES}
                    - --instance_type={RM_INSTANCE_TYPE}
                    - --gcs_scratch_location={GCS_SCRATCH_LOCATION}
                    - --enforce_kernel_ipv6_support=false
              volumes:
                - name: shared-memory
                  emptyDir:
                    medium: Memory
              serviceAccountName: default
              dnsPolicy: ClusterFirstWithHostNet

    # -------------------------------------------------------------------------
    # 2. Pathways Workers (TPU Pods)
    # -------------------------------------------------------------------------
    - name: pwwk
      replicas: {NUM_SLICES}
      template:
        metadata:
          annotations:
            alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
        spec:
          parallelism: {VMS_PER_SLICE}
          completions: {VMS_PER_SLICE}
          backoffLimit: 32
          template:
            spec:
              terminationGracePeriodSeconds: 60
              hostAliases:
                - ip: 169.254.169.254
                  hostnames:
                    - metadata
                    - metadata.google.internal
              nodeSelector:
                cloud.google.com/gke-tpu-accelerator: {GKE_ACCELERATOR}
                cloud.google.com/gke-tpu-topology: {GKE_TOPOLOGY}
{SPOT_NODE_SELECTOR_WORKER}
              tolerations:
                - key: google.com/tpu
                  operator: Equal
                  value: "present"
                  effect: NoSchedule
{SPOT_TOLERATION_WORKER}
{WORKER_INIT_CONTAINERS}
              containers:
                - name: worker
                  image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:jax-0.10.0
                  imagePullPolicy: Always
                  ports:
                    - containerPort: 8471
                    - containerPort: 8080
                    - containerPort: 8431
                    - containerPort: 9000
                    - containerPort: 29001
                  securityContext:
                    privileged: true
                  resources:
                    limits:
                      google.com/tpu: {CHIPS_PER_VM}
                  env:
                    - name: TPU_TYPE
                      value: {TPU_TYPE}
                    - name: NUM_TPU_SLICES
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                    - name: MEGASCALE_COORDINATOR_ADDRESS
                      value: {NAME}-pwhd-0-0.{NAME}
                    - name: MEGASCALE_NUM_SLICES
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
                    - name: MEGASCALE_SLICE_ID
                      valueFrom:
                        fieldRef:
                          fieldPath: metadata.labels['jobset.sigs.k8s.io/job-index']
                  args:
                    - --server_port=29001
                    - --resource_manager_address={NAME}-pwhd-0-0.{NAME}:29001
                    - --gcs_scratch_location={GCS_SCRATCH_LOCATION}
                    - --tpu_pinned_host_allocation_recycle=true
                    - --tpu_premapped_buffer_size={TPU_PREMAPPED_BUFFER_SIZE}
                    - --enforce_kernel_ipv6_support=false
{WORKER_VOLUME_MOUNTS}
              serviceAccountName: default
              dnsPolicy: ClusterFirstWithHostNet
{WORKER_VOLUMES}
  successPolicy:
    operator: All
    targetReplicatedJobs:
      - pwhd
"""
