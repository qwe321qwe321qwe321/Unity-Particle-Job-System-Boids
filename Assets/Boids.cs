//#define USE_OCT_TREE
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.ParticleSystemJobs;
#if USE_OCT_TREE
using NativeOctree;
#endif

[RequireComponent(typeof(ParticleSystem))]
public class Boids : MonoBehaviour {
    public int innerLoopCount = 256;

    public float separation = 0.1f;
    public float alignment = 0.5f;
    public float cohesion = 0.5f;
    public float visibleRadius = 10f;
    public float separationRadius = 2f;
    public float speed = 5f;

    public Vector3 size = new Vector3(30f, 20f, 20f);

    ParticleSystem m_ParticleSystem;
    BoidsJob m_BoidsJob;
    ApplyVelocitiesJob m_ApplyVelocitiesJob;
    NativeArray<float3> m_ParticleVelocities;

#if USE_OCT_TREE
    CacheJob m_CacheJob;
    OctreeJobs.AddBulkJob<int> m_AddBulkJob;
    NativeArray<OctElement<int>> m_OctElements;
    NativeOctree<int> m_OctTree;
#endif

    // Start is called before the first frame update
    void Start() {
        m_ParticleSystem = GetComponent<ParticleSystem>();
        int maxParticleCount = m_ParticleSystem.main.maxParticles;
        m_ParticleVelocities = new NativeArray<float3>(maxParticleCount, Allocator.Persistent);

#if USE_OCT_TREE
        m_OctElements = new NativeArray<OctElement<int>>(maxParticleCount, Allocator.Persistent);
        m_OctTree = new NativeOctree<int>(new AABB() { Center = Vector3.zero, Extents = size }, Allocator.Persistent);
        m_CacheJob = new CacheJob() {
            Elements = m_OctElements
        };
        m_AddBulkJob = new OctreeJobs.AddBulkJob<int>() {
            Elements = m_OctElements,
            Octree = m_OctTree,
        };
#endif
        m_BoidsJob = new BoidsJob() {
            separation = separation,
            alignment = alignment,
            cohesion = cohesion,
            visibleRadius = visibleRadius,
            separationRadius = separationRadius,
            speed = speed,
            appliedVelocities = m_ParticleVelocities,

#if USE_OCT_TREE
            indexOctTree = m_OctTree,
#endif
        };

        m_ApplyVelocitiesJob = new ApplyVelocitiesJob() {
            applidedVelocities = m_ParticleVelocities
        };

    }

    private void OnDestroy() {
        m_ParticleVelocities.Dispose();
#if USE_OCT_TREE
        m_OctElements.Dispose();
        m_OctTree.Dispose();
#endif
    }

    private void Update() {
        m_BoidsJob.deltaTime = Time.deltaTime;
        float3 halfSize = size / 2;
        m_BoidsJob.lowerBound = -halfSize;
        m_BoidsJob.upperBound = halfSize;

        m_BoidsJob.separation = separation;
        m_BoidsJob.alignment = alignment;
        m_BoidsJob.cohesion = cohesion;
        m_BoidsJob.visibleRadius = visibleRadius;
        m_BoidsJob.separationRadius = separationRadius;
        m_BoidsJob.speed = speed;
    }

    private void OnParticleUpdateJobScheduled() {
#if USE_OCT_TREE
        var handle = m_CacheJob.ScheduleBatch(m_ParticleSystem, innerLoopCount);
        handle = m_AddBulkJob.Schedule(handle);
        handle = m_BoidsJob.ScheduleBatch(m_ParticleSystem, innerLoopCount, handle);
        handle = m_ApplyVelocitiesJob.ScheduleBatch(m_ParticleSystem, innerLoopCount, handle);
#else
        var handle = m_BoidsJob.ScheduleBatch(m_ParticleSystem, innerLoopCount);
        handle = m_ApplyVelocitiesJob.ScheduleBatch(m_ParticleSystem, innerLoopCount, handle);
#endif
    }

    private void OnDrawGizmos() {
        Gizmos.DrawWireCube(this.transform.position, size);
    }


#if USE_OCT_TREE
    [BurstCompile]
    struct CacheJob : IJobParticleSystemParallelForBatch {
        [WriteOnly]
        public NativeArray<OctElement<int>> Elements;

        public void Execute(ParticleSystemJobData particles, int startIndex, int count) {
            var srcPositions = particles.positions;

            int endIndex = startIndex + count;
            for (int i = startIndex; i < endIndex; i++)
                Elements[i] = new OctElement<int> { element = i, pos = srcPositions[i] };
        }
    }
#endif

    [BurstCompile]
    struct BoidsJob : IJobParticleSystemParallelForBatch {
        public float separation;
        public float alignment;
        public float cohesion;
        public float visibleRadius;
        public float separationRadius;

        public float speed;
        public float deltaTime;
        public float3 lowerBound;
        public float3 upperBound;

#if USE_OCT_TREE
        [ReadOnly]
        public NativeOctree<int> indexOctTree;
#endif

        [WriteOnly]
        public NativeArray<float3> appliedVelocities;

        public void Execute(ParticleSystemJobData particles, int startIndex, int count) {
            var positions = particles.positions;
            var velocities = particles.velocities;

            int endIndex = startIndex + count;
            int paritcleCount = particles.count;

            float visibleRadiusSquared = visibleRadius * visibleRadius;
            float separationRadiusSquared = separationRadius * separationRadius;
            for (int i = startIndex; i < endIndex; i++) {
                float3 position = positions[i];
                float3 velocity = velocities[i];
                AABB particleAABB = new AABB() {
                    Center = position,
                    Extents = new float3(visibleRadius, visibleRadius, visibleRadius)
                };


                float3 sumPosition = float3.zero;
                float3 sumVelocity = float3.zero;
                float3 separationVelocity = float3.zero;
                int countInRange = 0;

#if USE_OCT_TREE
                NativeList<OctElement<int>> queryResultList = new NativeList<OctElement<int>>(Allocator.TempJob);

                indexOctTree.RangeQuery(particleAABB, queryResultList);
                for (int q = 0; q < queryResultList.Length; q++) {
                    int index = queryResultList[q].element;
                    if (i == index) { continue; }
                    float3 otherPosition = positions[index];
                    float3 otherVelocity = velocities[index];
                    // in range.
                    float distanceSquared = math.distancesq(position, otherPosition);
                    if (distanceSquared < visibleRadiusSquared) {
                        sumPosition += otherPosition;
                        sumVelocity += otherVelocity;
                        if (distanceSquared < separationRadiusSquared) {
                            float3 translationDirection = math.normalize(position - otherPosition);
                            float distance = math.sqrt(distanceSquared);
                            separationVelocity += (separationRadius - distance) * translationDirection;

                        }
                        countInRange += 1;
                    }
                }
                queryResultList.Dispose();
#else
                for (int j = 0; j < paritcleCount; j++) {
                    if (i == j) { continue; }
                    float3 otherPosition = positions[j];
                    float3 otherVelocity = velocities[j];
                    // in range.
                    float distanceSquared = math.distancesq(position, otherPosition);
                    if (distanceSquared < visibleRadiusSquared) {
                        sumPosition += otherPosition;
                        sumVelocity += otherVelocity;
                        if (distanceSquared < separationRadiusSquared) {
                            float3 translationDirection = math.normalize(position - otherPosition);
                            float distance = math.sqrt(distanceSquared);
                            separationVelocity += (separationRadius - distance) * translationDirection;

                        }
                        countInRange += 1;
                    }
                }
#endif


                float3 acceleration = float3.zero;
                if (countInRange > 0) {
                    sumPosition /= countInRange;
                    sumVelocity /= countInRange;
                    //separationVelocity /= countInRange;

                    float currentSpeed = math.length(velocity);
                    float3 averageDirection = math.normalize(sumVelocity);
                    float3 alignmentForce = alignment * (currentSpeed * averageDirection - velocity);
                    float3 cohesionDirection = math.normalize(sumPosition - position);
                    float3 cohesionForce = cohesion * (currentSpeed * cohesionDirection - velocity);
                    //float3 separationDirection = math.normalize(separationVelocity);
                    //float3 separationForce = separation * (currentSpeed * separationDirection - velocity);
                    float3 separationForce = separation * separationVelocity;
                    acceleration += alignmentForce + cohesionForce + separationForce;
                }
                velocity += acceleration * deltaTime;

                if ((position.x <= lowerBound.x && velocity.x < 0f) ||
                    (position.x >= upperBound.x && velocity.x > 0f)) {
                    velocity.x = -velocity.x;
                }
                if ((position.y <= lowerBound.y && velocity.y < 0f) ||
                    (position.y >= upperBound.y && velocity.y > 0f)) {
                    velocity.y = -velocity.y;
                }
                if ((position.z <= lowerBound.z && velocity.z < 0f) ||
                   (position.z >= upperBound.z && velocity.z > 0f)) {
                    velocity.z = -velocity.z;
                }

                velocity = math.max(speed, math.length(velocity)) * math.normalize(velocity);
                appliedVelocities[i] = velocity;
            }
        }
    }

    struct ApplyVelocitiesJob : IJobParticleSystemParallelForBatch {
        [ReadOnly]
        public NativeArray<float3> applidedVelocities;
        public void Execute(ParticleSystemJobData particles, int startIndex, int count) {
            var velocities = particles.velocities;

            int endIndex = startIndex + count;

            for (int i = startIndex; i < endIndex; i++) {
                velocities[i] = applidedVelocities[i];
            }
        }
    }
}
