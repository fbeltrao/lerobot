import huggingface_hub

repo_id = "fbeltrao/so101_unplug_cable_4"


tag = "v2.1"
source_branch_to_tag = "main"
huggingface_hub.delete_tag(repo_id=repo_id, tag=tag, repo_type="dataset")
huggingface_hub.create_tag(repo_id=repo_id, tag=tag, repo_type="dataset", revision=source_branch_to_tag)
print(f"Created tag {tag} for {repo_id} from branch {source_branch_to_tag}")


# Delete a specific branch
# branch_to_delete = "v2.1"
# huggingface_hub.delete_branch(repo_id=repo_id, branch=branch_to_delete, repo_type="dataset")
# print(f"Deleted branch {branch_to_delete} from {repo_id}")


# Delete specific files from the dataset
# for i in range(54, 60):
#     file = f"/data/chunk-000/episode_{i:06d}.parquet"

#     huggingface_hub.delete_file(
#         repo_id=repo_id, path_in_repo=file, repo_type="dataset", revision=source_branch_to_tag
#     )
